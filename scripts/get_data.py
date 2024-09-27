import datetime

import pytz
from forecast_models.data_sources.time_series_api import TimeSeriesAPIClient, APIEnvironment

tsc = TimeSeriesAPIClient(environment=APIEnvironment.PRODUCTION)

start_time = datetime.datetime(2023, 2, 15, 16, 0, 0, tzinfo=pytz.UTC)
end_time = datetime.datetime(2023, 6, 15, 16, 0, 0, tzinfo=pytz.UTC)
data = tsc.get_timeseries_data(
    site_name="samohi",
    gateway_id="2b0f7ee2adf64b458ecbcfb28213851e",
    load_types=["pv"],
    start_time=start_time,
    end_time=end_time,
)

data.to_csv("/home/jackie/etb/model-performance-tracking/data/unseen_pv.csv")
