import pathlib

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder

from aiconf.reader import BBCReader
from aiconf.model import BBCModel

FIXTURES_ROOT = pathlib.Path(__file__).parent / 'fixtures'


class PaperModelTest(ModelTestCase):
    def setUp(self):
        super().setUp()

        self.set_up_model(FIXTURES_ROOT / 'experiment.json', FIXTURES_ROOT / 'tiny.csv')

        # 1. call self.set_up_model with the path to the experiment config
        #    and the path to the test fixture

    def test_simple_tagger_can_train_save_and_load(self):
        # Doesn't work on Windows: 
        # "PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: ..."
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_forward_pass_runs_correctly(self):
        # feel free to add extra tests here
        pass
