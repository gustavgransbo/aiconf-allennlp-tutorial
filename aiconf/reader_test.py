import pathlib

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers import SingleIdTokenIndexer

from aiconf.reader import BBCReader

FIXTURES_ROOT = pathlib.Path(__file__).parent / 'fixtures'

CATEGORIES = {'tech', 'sport', 'politics', 'entertainment', 'business'}

class ReaderTest(AllenNlpTestCase):
    def test_n_instances(self):
        token_indexers = {"tokens": SingleIdTokenIndexer()}
        reader = BBCReader(token_indexers)
        instances = reader.read(str(FIXTURES_ROOT / 'tiny.csv'))

        assert len(instances) == 5

    def test_labels(self):
        token_indexers = {"tokens": SingleIdTokenIndexer()}
        reader = BBCReader(token_indexers)
        instances = reader.read(str(FIXTURES_ROOT / 'tiny.csv'))

        assert not set(instance['category'].label for instance in instances).difference(CATEGORIES)

    def test_instance_values(self):
        token_indexers = {"tokens": SingleIdTokenIndexer()}
        reader = BBCReader(token_indexers)
        instances = reader.read(str(FIXTURES_ROOT / 'tiny.csv'))
        first_instance = instances[0]
        assert [token.text for token in first_instance['text'].tokens] == [
            "Computer", "grid", "to", "help", "the", "world", 
            "Your", "computer", "can", "now", "help", "solve",
            "the", "worlds", "most", "difficult", "health", "and", "social", "problems"
        ]

        assert first_instance['category'].label == 'tech'

        # Some ideas of things to test:
        # * test that there are 5 instances
        # * test that there's one of each label
        # * test that the first instance has the right values in its fields
