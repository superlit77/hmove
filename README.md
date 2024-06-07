# Introduction
This document is included in the 'Three Heads Are Better Than One: Suggesting Move Method Refactoring Opportunities with Inter-class Code Entity Dependency Enhanced Hypergraph Neural Network' distribution, which we will refer to as HMove. This is to distinguish the recommended implementation of this Move Method refactoring from other implementations. In this document, the environment required to make and use the HMove tool is described. Some hints about the installation environment are here, but users need to find complete instructions from other sources. They give a more detailed description of their tools and instructions for using them. Our main environment is located on a computer with windows (windows 11) operating system. The fundamentals should be similar for other platforms, although the way in which the environment is configured will be different. What do I mean by environment? For example, to run python code you will need to install a python interpreter, and if you want to use pre-trained model you will need torch.

# HMove
/src: The code files which is involved in the experiment \
/data_demo: relevant data of the example involved in Section 2 of the paper \
/RQ3: the questionnaire and case study results \
/tool:  a Visual Studio Code (VSCode) extension of hmove

# Technique
## pre-trained model
CodeBERT GraphCodeBERT CodeGPT CodeT5 CodeT5+ CodeTrans

# Requirement
## CodeBERT, GraphCodeBERT, CodeGPT, CodeT5, CodeT5+
python3(>=3.6) \
we use python 3.9\
torch transformers \
we use torch(1.13.1) and transformers(4.20.1)\
pre-trained model link: \
CodeBERT: https://huggingface.co/microsoft/codebert-base \
CodeGPT: https://huggingface.co/microsoft/CodeGPT-small-java-adaptedGPT2 \
GraphCodeBERT: https://huggingface.co/microsoft/graphcodebert-base \
CodeT5: https://huggingface.co/Salesforce/codet5-base-multi-sum \
CodeT5+: https://huggingface.co/Salesforce/codet5p-6b \

CodeTrans:https://huggingface.co/SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune \

## hyper-parameter settings

| Embedding Technique |                   Hyper-parameter settings                   |
| :-----------------: | :----------------------------------------------------------: |
|      CodeBERT       | train\_batch\_size=2048, embeddings\_size =768, learning\_rate=5e-4, max\_position\_length=512 |
|    GraphCodeBERT    | train\_batch\_size=1024, embeddings\_size =768, learning\_rate=2e-4, max\_sequence\_length=512 |
|       CodeGPT       |      embeddings\_size =768, max\_position\_length=1024       |
|       CodeT5        | train\_batch\_size=1024, embeddings\_size =768, learning\_rate=2e-4, max\_sequence\_length=512 |
|       CodeT5+       | train\_batch\_size=2048, embeddings\_size =256, learning\_rate=2e-4, max_sequence_length=512 |
|      CodeTrans      | train\_batch\_size=4096, embeddings\_size =768, learning\_rate=2e-4, max_sequence_length=512 |

# Quickstart

##  Training phase

> step1: we extract inter-class entity dependency graphs from both training samples and transform these into inter-class entity dependency hypergraphs.

> step 2: nodes in these hypergraphs are assigned with attributes with pre-trained code model.

> step 3: these attributed inter-class entity dependency hypergraphs are fed into an enhanced hypergraph neural network for the purpose of training.

##  Detection phase

> The detection phase extracts attributed hypergraphs through hypergraph construction and entity attribute generation. Taking attributed hypergraphs as input, the detection phase further conducts refactoring opportunity suggestions via trained model invocation and LLM-based precondition verification.

# Datasets

AccEval: [Terra et al's dataset](http://java.llp.dcc.ufmg.br/jmove/) 

HumanEval:  [JFreeChart](https://github.com/jfree/jfreechart), [JGroups](https://github.com/belaban/JGroups),[Derby](https://github.com/apache/derby),[DrJava ](https://github.com/DrJavaAtRice/drjava),[JTOpen](https://github.com/IBM/JTOpen),[Ant](https://github.com/apache/ant),[Tapestry](https://github.com/apache/tapestry-5),[lucene](https://github.com/apache/lucene),[mvnForum](https://github.com/khanhnguyenj/mvnForumJ).
