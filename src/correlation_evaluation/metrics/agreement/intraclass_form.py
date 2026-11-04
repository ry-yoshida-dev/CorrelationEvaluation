from enum import Enum


class IntraclassForm(Enum):
    """
    Study design of an intraclass correlation, in Shrout and Fleiss's terms.

    Attributes
    ----------
    ONE_WAY_RANDOM
        ICC(1,1): each subject has its own random raters.
    TWO_WAY_RANDOM
        ICC(2,1): absolute agreement, with raters sampled from a population.
    TWO_WAY_MIXED
        ICC(3,1): consistency, with these raters being the only ones of interest.
    """

    ONE_WAY_RANDOM = "one_way_random"
    TWO_WAY_RANDOM = "two_way_random"
    TWO_WAY_MIXED = "two_way_mixed"
