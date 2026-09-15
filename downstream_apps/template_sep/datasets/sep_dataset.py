import tempfile
import numpy as np
import pandas as pd
from typing import Literal
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset


class SEPDSDataset(HelioNetCDFDataset):
    """
    Downstream dataset for Solar Energetic Particle (SEP) binary classification.

    Extends ``HelioNetCDFDataset`` with a binary SEP label (0.0 / 1.0) aligned
    to the Surya index via temporal matching. The label is returned as ``np.float32``
    so it is compatible with ``BCELoss`` / sigmoid heads directly.

    The SEP CSV (``ds_sep_index_path``) is used as **both** the Surya index and the
    label source when no separate ``index_path`` is supplied. It must have at minimum:

        timestep                path                     SEP
        2017-09-10 00:00:00     s3://bucket/...           1
        2015-03-20 00:00:00     s3://bucket/...           1
        2019-09-05 00:00:00     s3://bucket/...           0
        ...

    ``HelioNetCDFDataset`` requires a ``present`` column in its index. If the CSV
    lacks one, this class synthesises it — all rows are marked ``present=1`` since
    every row in the SEP CSV is an intentionally-labelled sample. A temp file is
    written transparently; the original CSV is never modified.

    All ``HelioNetCDFDataset`` keyword arguments (``index_path``, ``scalers``,
    ``channels``, ``s3_cache_dir``, etc.) are accepted via ``**kwargs`` and
    forwarded to the base class. ``load_forecast_frames`` defaults to ``False``
    because SEP forecasting supplies its own labels.

    Args:
        return_surya_stack: If ``True`` (default), include the Surya image stack
            in the returned dict. Set to ``False`` to return only the SEP label
            (useful for label inspection or fast iteration).
        max_number_of_samples: Cap the dataset length at this value. ``None``
            means use all matched samples.
        ds_sep_index_path: Path to the SEP label CSV. Required. Used as
            ``index_path`` for the parent if no separate ``index_path`` is given.
        ds_time_column: Column in the SEP CSV used as the event timestamp.
            Defaults to ``"timestep"``.
        ds_time_tolerance: Maximum allowed gap when matching SEP and Surya
            timestamps (e.g., ``"4d"``). Unmatched rows are dropped. Pass
            ``None`` to skip tolerance filtering.
        ds_match_direction: Merge direction passed to ``pd.merge_asof``.
            ``"forward"`` (default) is causal — uses the solar state *before*
            the SEP event.

    Raises:
        ValueError: If ``ds_sep_index_path`` is not provided, if the SEP column
            is missing from the CSV, or if no timestamps overlap between the
            Surya and SEP indices within the given tolerance.
    """

    def __init__(
        self,
        # Downstream-specific parameters
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        ds_sep_index_path: str | None = None,
        ds_time_column: str = "timestep",
        ds_time_tolerance: str | None = None,
        ds_match_direction: Literal["forward", "backward", "nearest"] = "forward",
        # All HelioNetCDFDataset parameters forwarded transparently
        **kwargs,
    ):
        if ds_match_direction not in ("forward", "backward", "nearest"):
            raise ValueError(
                "ds_match_direction must be one of 'forward', 'backward', or 'nearest'"
            )
        if ds_sep_index_path is None:
            raise ValueError("ds_sep_index_path must be provided for SEPDSDataset")

        # ------------------------------------------------------------------
        # If no separate index_path was given, use the SEP CSV as the Surya
        # index too. HelioNetCDFDataset requires a "present" column; synthesise
        # it (all 1s — every labelled row is a valid sample) into a temp file
        # so the original CSV is never modified.
        # ------------------------------------------------------------------
        if "index_path" not in kwargs:
            raw = pd.read_csv(ds_sep_index_path)
            if "present" not in raw.columns:
                raw["present"] = 1
            # NamedTemporaryFile with delete=False: the parent reads it during
            # __init__; we clean it up immediately after super().__init__() returns.
            self._tmp_index = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            )
            raw.to_csv(self._tmp_index.name, index=False)
            self._tmp_index.close()
            kwargs["index_path"] = self._tmp_index.name
        else:
            self._tmp_index = None

        # Flare forecasting supplies its own labels, so future Surya frames are
        # never needed — mirror the same default as FlareDSDataset.
        kwargs.setdefault("load_forecast_frames", False)
        try:
            super().__init__(**kwargs)
        finally:
            # Always clean up the temp file, even if the parent raises.
            if self._tmp_index is not None:
                import os
                try:
                    os.unlink(self._tmp_index.name)
                except OSError:
                    pass

        self.return_surya_stack = return_surya_stack

        # ------------------------------------------------------------------
        # Load and validate the SEP index
        # ------------------------------------------------------------------
        self.ds_index = pd.read_csv(ds_sep_index_path)

        if "SEP" not in self.ds_index.columns:
            raise ValueError(
                f"The SEP index CSV at '{ds_sep_index_path}' must contain a 'SEP' column. "
                f"Found columns: {self.ds_index.columns.tolist()}"
            )

        self.ds_index["ds_index"] = (
            pd.to_datetime(self.ds_index[ds_time_column])
            .values.astype("datetime64[ns]")
        )
        self.ds_index.sort_values("ds_index", inplace=True)

        # Cast label to float32 now so __getitem__ needs no per-sample work.
        self.ds_index["sep_label"] = self.ds_index["SEP"].astype(np.float32)

        # ------------------------------------------------------------------
        # Align SEP timestamps to the nearest valid Surya timestep
        # ------------------------------------------------------------------
        df_valid = (
            pd.DataFrame({"valid_indices": self.valid_indices})
            .sort_values("valid_indices")
        )
        df_valid = pd.merge_asof(
            df_valid,
            self.ds_index[["ds_index", "sep_label"]],
            right_on="ds_index",
            left_on="valid_indices",
            direction=ds_match_direction,
        )

        # Keep the closest Surya frame per SEP event (handles many-to-one matches)
        df_valid["index_delta"] = np.abs(
            df_valid["valid_indices"] - df_valid["ds_index"]
        )
        df_valid = df_valid.sort_values(["ds_index", "index_delta"])
        df_valid.drop_duplicates(subset="ds_index", keep="first", inplace=True)

        # Drop rows that did not match any SEP event
        df_valid = df_valid.dropna(subset=["ds_index"])

        # Enforce maximum time tolerance
        if ds_time_tolerance is not None:
            df_valid = df_valid.loc[
                df_valid["index_delta"] <= pd.Timedelta(ds_time_tolerance)
            ]
            if len(df_valid) == 0:
                raise ValueError(
                    "No overlap found between the Surya index and the SEP index "
                    f"within the tolerance '{ds_time_tolerance}'. "
                    "Try widening ds_time_tolerance or checking the timestamp format."
                )

        # ------------------------------------------------------------------
        # Override parent valid-index state with matched subset
        # ------------------------------------------------------------------
        self.valid_indices = [
            pd.Timestamp(ts) for ts in df_valid["valid_indices"]
        ]
        self.adjusted_length = len(self.valid_indices)
        df_valid.set_index("valid_indices", inplace=True)
        self.df_valid_indices = df_valid

        if (
            max_number_of_samples is not None
            and max_number_of_samples < self.adjusted_length
        ):
            self.valid_indices = self.valid_indices[:max_number_of_samples]
            self.df_valid_indices = self.df_valid_indices.iloc[:max_number_of_samples]
            self.adjusted_length = max_number_of_samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return a single sample.

        Args:
            idx: Dataset index.

        Returns:
            Dictionary containing:

            ``sep`` (np.float32):
                Binary SEP label — ``1.0`` if an SEP event was recorded,
                ``0.0`` otherwise. Ready for ``BCELoss`` with a sigmoid head.
            ``ds_index`` (str):
                ISO-format timestamp from the SEP index, for bookkeeping.

            When ``return_surya_stack=True`` (default), all keys from
            ``HelioNetCDFDataset.__getitem__`` are also present
            (``ts``, ``time_delta_input``, and, if ``load_forecast_frames=True``,
            ``forecast`` and ``lead_time_delta``).
        """
        sample = super().__getitem__(idx=idx) if self.return_surya_stack else {}
        row = self.df_valid_indices.iloc[idx]
        sample["sep"] = row["sep_label"]                      # np.float32, 0.0 or 1.0
        sample["ds_index"] = row["ds_index"].isoformat()      # e.g. "2017-09-10T00:00:00"
        return sample