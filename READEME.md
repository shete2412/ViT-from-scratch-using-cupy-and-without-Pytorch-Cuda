#Core Component 01 (Attention Implementation)
Implement a vision transformer architecture and train it end-to-end for classification on the provided dataset. The
vision transformer architecture must consist of at least:
• Image tokenizer
• Two transformer block with multi-head self attention. Each transformer block must consist at least:– Normalization layer– Skip residual connection– Multi-layer perceptron– Query, Key and Value projection– At least 2 heads in multi-head self attention

The complete architecture must be trainable end-to-end. Students are responsible for implementing the forward
and backward computations required to train their models. The implementation must not rely on pre-existing
implementations of Transformer layers, attention layers, or automatic differentiation.

![Architecture](architecture\architecture.png)
![Gradients Flow](architecture\backward_gradients.png)

#Core Component 2 (Model Scaling)
In this experiment, we will investigate the effect of model capacity while keeping the training dataset fixed. we
must conduct controlled experiments, where only the intended architectural parameter(s) are changed while other
experimental conditions are kept as consistent as possible.
• Number of layers
• Attention heads
• Embedding dimension
• and so on...
we should analyse the impact of scaling the model using any of the hyperparameter and derive a conclusive
impact on the performance in terms of accuracy vs computation. For each experiment, report:
• Number of parameters
• FLOPs
• MACs
• Training and validation performance
• Inference latency
• Inference throughput

Experiments:
  01. baseline           | L=3 H=4 D=256 MLP=4
  02. layers_2           | L=2 H=4 D=256 MLP=4
  03. layers_4           | L=4 H=4 D=256 MLP=4
  04. layers_6           | L=6 H=4 D=256 MLP=4
  05. heads_2            | L=3 H=2 D=256 MLP=4
  06. heads_8            | L=3 H=8 D=256 MLP=4
  07. heads_16           | L=3 H=16 D=256 MLP=4
  08. embed_128          | L=3 H=4 D=128 MLP=4
  09. embed_192          | L=3 H=4 D=192 MLP=4
  10. embed_384          | L=3 H=4 D=384 MLP=4
  11. mlp_ratio_2        | L=3 H=4 D=256 MLP=2
  12. mlp_ratio_6        | L=3 H=4 D=256 MLP=6

![Baseline Accuracy](core_component2_results\plots\baseline_accuracy_history.png)
![Layer 2 Accuracy](core_component2_results\plots\layers_2_accuracy_history.png)


#Core Component 3 (Data Scaling)
In this experiment, we will investigate the effect of training-data size while keeping the Transformer architecture
fixed. we will train the same baseline architecture using different amounts of the provided training data. For
example: 10%,25%,50%,100%. Explore the impact of uniform sampling of data vs non uniform sampling as well
and its impact on classification score.
For each data scale, report:
• Training time
• Training and validation performance

![performance non-uniform 10% training data](core_component3_results\plots\history_scores_nonuniform_010pct.png)
![performance non-uniform 25% training data](core_component3_results\plots\history_scores_nonuniform_025pct.png)
![performance uniform 10% training data](core_component3_results\plots\history_scores_uniform_010pct.png)
![performance uniform 25% training data](core_component3_results\plots\history_scores_uniform_025pct.png)

