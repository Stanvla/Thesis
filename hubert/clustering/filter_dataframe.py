from abc import abstractmethod, ABC
from warnings import warn


class Filter(ABC):
    def __init__(self, value, name):
        self.value = value
        self.name = name

    def check_name(self, df):
        if self.name not in df.columns:
            raise RuntimeError(f'Can not filter the dataframe using column `{self.name}` since it is not in {df.columns}')

    def check_empty(self, df):
        if len(df) == 0:
            warn(f'The dataframe is empty after filtering {self.name} with value {self.value} using filter class {self.__class__.__name__}.')

    @abstractmethod
    def __call__(self, df):
        ...


class FilterLB(Filter):
    def __init__(self, value, name):
        super(FilterLB, self).__init__(value, name)

    def __call__(self, df):
        self.check_name(df)
        new_df = df[df[self.name] >= self.value]
        self.check_empty(new_df)
        return new_df


class FilterUB(Filter):
    def __init__(self, value, name):
        super(FilterUB, self).__init__(value, name)

    def __call__(self, df):
        self.check_name(df)
        new_df = df[df[self.name] <= self.value]
        self.check_empty(new_df)
        return new_df


def clean_data_parczech(df, filer_list):
    for f in filer_list:
        df = f(df)
    return df
