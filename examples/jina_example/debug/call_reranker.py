from collections import namedtuple
import os
import faiss
from rocket_qa.model.src import rocket_rerank
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc
import paddle
paddle.enable_static()


args = {
    'use_cuda': False,
    'batch_size':128,
    'init_checkpoint_dir':'rocket_qa/checkpoint/marco_cross_encoder_large'
}

# Args = namedtuple('Args',args)
# args = Args(**args)

# data = ["what is paula deen's brother	-	-	0","hello1	-	-	0","hello2	-	-	0"]
# data = ["what is paula deen's brother	-	-	0"]
data = ["what is paula deen's brother\t-\tThe presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.\t0"]
# data = ["-\t-\tThe presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.\t0"]
# inference_de.main(args)
reranker = rocket_rerank.RocketReRanker(**args)
scores = reranker.get_scores(data)
print(scores)
# print(scores)
print(scores.shape)
