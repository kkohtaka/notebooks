from datetime import datetime, timedelta, timezone
import click
import pandas as pd

from prometheus import Prometheus


@click.command()
@click.option(
    '--prometheus-server',
    default='http://localhost:9090',
    help='Host name of Prometheus server, including sheme and port number',
)
@click.option(
    '--output',
    default='output.csv',
    help='Path to an output CSV file',
)
@click.option(
    '--start-date',
    default='2018/01/01',
    help='Start date of data to fetch',
)
def fetch(
    prometheus_server,
    output,
    start_date,
):
    p = Prometheus(prometheus_server)

    header_written = False
    start = datetime.strptime(
        start_date, '%Y/%m/%d').replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    with open(output, 'a') as f:
        # Truncate existing data
        f.truncate(0)

        while start < now:
            duration = timedelta(days=1)
            step = timedelta(minutes=1)
            data = p.range_query(
                'max(bitflyer_last_traded_price{product_code="BTC_JPY"}) by (product_code)',
                start,
                duration,
                step,
            )

            if len(data) > 0:
                series = pd.Series(
                    data[0]['values'].T[1],
                    index=data[0]['values'].T[0],
                )
                df = series.to_frame(name='ltp')
                df.index = pd.to_datetime(df.index, unit='s')
                df.index.name = 'timestamp'

                df.to_csv(f, header=(not header_written))
                header_written = True

            start += duration


if __name__ == '__main__':
    fetch()
