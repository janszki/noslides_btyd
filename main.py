import pandas as pd
from btyd import BetaGeoFitter, GammaGammaFitter
from btyd.plotting import (
    plot_calibration_purchases_vs_holdout_purchases,
    plot_period_transactions,
)
from btyd.utils import calibration_and_holdout_data


def load_CDNOW_master():
    return pd.read_csv("CDNOW_master_cleaned.csv", parse_dates=["date"])


def load_CDNOW_sample():
    return pd.read_csv("CDNOW_sample_cleaned.csv", parse_dates=["date"])


if __name__ == "__main__":
    calibration_period_end = "1997-12-31"
    observation_period_end = "1998-06-30"

    transaction_data = load_CDNOW_master()

    # split into calibration/holdout phases and summarize transactions into
    # Recency, Frequency, and Monetary value
    summary_cal_holdout = calibration_and_holdout_data(
        transaction_data,
        "customer_id",
        "date",
        calibration_period_end=calibration_period_end,
        observation_period_end=observation_period_end,
        monetary_value_col="value",
    )

    #
    # frequency modeling
    #
    bgf = BetaGeoFitter().fit(
        summary_cal_holdout["frequency_cal"],
        summary_cal_holdout["recency_cal"],
        summary_cal_holdout["T_cal"],
    )

    # how does model do at reproducing calibration period frequencies?
    plot_period_transactions(bgf, color=["#000000", "#999999"])

    # how does model do at predicting holdout period frequencies?
    plot_calibration_purchases_vs_holdout_purchases(
        bgf, summary_cal_holdout, n=9, color=["#000000", "#999999"]
    )

    #
    # transaction value modeling
    #
    gg_model = GammaGammaFitter()

    # we only train on customers with multiple purchases in the calibration phase
    df_repeats = summary_cal_holdout[summary_cal_holdout["monetary_value_cal"] > 0]

    # fit the model with calibration period data
    gg_model.fit(df_repeats["frequency_cal"], df_repeats["monetary_value_cal"])

    # model parameters learned above are used to compare model and actual avg monetary values
