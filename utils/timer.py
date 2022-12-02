from timeit import default_timer


class elapsed_timer(object):

    def __init__(self,
                 title: str):

        self._title = title

    def __enter__(self):

        print(self._title, end=' ', flush=True)
        self._start = default_timer()

    def __exit__(self,
                 ex_type,
                 val,
                 traceback):

        self._end = default_timer()
        print('takes {:.2f}s'.format(self._end - self._start))