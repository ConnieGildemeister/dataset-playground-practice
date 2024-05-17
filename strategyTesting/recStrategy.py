import abc

class RecStrategy(abc.ABC):
    """
    The base class for the various recommendation strategies.
    Each strategy should have support for recommending artists and events.

    **kwargs is used as the parameters for now until the data required is determined.

    Likely needs:
    num (how many are being recommended)
    query (indecies of data)
    model (the model used for conversion)
    vector (the vectorized data)
    data (the table of artists/events)
    """

    @abc.abstractmethod
    def recommend_artist(self, **kwargs):
        pass

    @abc.abstractmethod
    def recommend_event(self, **kwargs):
        pass