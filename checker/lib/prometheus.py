from urllib import request, parse
import json
import numpy as np


class Prometheus:

    PROMETHEUS_RANGE_QUERY_PATH = '/api/v1/query_range'

    def __init__(self, host):
        self.host = host

    def range_query(self, query, start, duration, step):
        """Sends a request to Prometheus Range Query API and returns a response in NumPy format.
        Args:
            query (str): query written in PromQL
            start (datetime.datetime): start of query time range
            duration (datetime.timedelta): width of query time range
            step (datetime.timedelta): query resolution step width
        Returns:
            list of query results
        """
        params = {}
        params["query"] = query
        params["start"] = start.timestamp()
        params["end"] = (start + duration).timestamp()
        params["step"] = '%ds' % step.total_seconds()
        query = parse.urlencode(params)

        url = "%s%s?%s" % (self.host, self.PROMETHEUS_RANGE_QUERY_PATH, query)

        res = []
        req = request.Request(url, method="GET")
        with request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            metrics = data["data"]["result"]
            for metric in metrics:
                res.append({
                    'metric': metric['metric'],
                    'values': np.array(metric['values'], dtype=np.int64),
                })
        return res
