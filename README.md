# Normative Evaluation of Large Language Models with Everyday Moral Dilemmas
by Pratik Sachdeva and Tom van Nuenen

# Overview
This repository provides the code to reproduce the analyses and figures
developed in the paper *Normative Evaluation of Large Language Models with Everyday Moral Dilemmas*
by Sachdeva & van Nuenen, soon to be published in the 2025 Fairness, Accountability, and Transparency (FAccT).

# Abstract

The rapid adoption of large language models (LLMs) has spurred extensive research into their encoded moral norms and decision-making processes. Much of this research relies on prompting LLMs with survey-style questions to assess how well models are aligned with certain demographic groups, moral beliefs, or political ideologies. While informative, the adherence of these approaches to relatively superficial constructs, beliefs, and moral questions tends to oversimplify the complexity and nuance underlying everyday moral dilemmas. We argue that auditing LLMs along more detailed axes of human interaction is of paramount importance to better assess the degree to which they may impact human beliefs and actions. To this end, we evaluate LLMs on complex, everyday moral dilemmas sourced from the ``Am I the Asshole" (AITA) community on Reddit, where users seek moral judgments on everyday conflicts from other community members. We prompted seven commonly used LLMs, including proprietary and open-source models, to assign blame and provide explanations for over 10,000 AITA moral dilemmas. We then compared the LLMs' judgments and explanations to those of Redditors and to each other, aiming to uncover patterns in their moral reasoning. Our results demonstrate that large language models exhibit distinct patterns of moral judgment, varying substantially from human evaluations on the AITA subreddit. LLMs demonstrate moderate to high self-consistency but low inter-model agreement, suggesting that differences in training and alignment lead to fundamentally different approaches to moral reasoning. We further observe that an ensemble of LLMs, despite individual inconsistencies, collectively approximates Redditor consensus in assigning blame. Further analysis of model explanations reveals distinct patterns in how models invoke various moral principles, with some models showing greater sensitivity to specific themes such as fairness or harm. These findings highlight the complexity of implementing consistent moral reasoning in artificial systems and the need for careful evaluation of how different models approach ethical judgment. As LLMs continue to be used in roles requiring ethical decision-making such as therapists and companions, careful evaluation is crucial to mitigate potential biases and limitations. Despite the capacity of LLMs to analyze moral dilemmas, their judgments ultimately lack the ethical accountability of human deliberation, requiring careful scrutiny and reflection on their role in ethical discourse.

## Repository Structure

The repository is divided into the following folders:

* `figures`: Jupyter notebooks used to generate the figures in the paper.
* `src`: Contains the codebase used in scripts, analyses, and figure
  generation for this paper.
* `notebooks`: Contains supplementary Jupyter notebooks used in secondary
  analyses.
* `scripts`: Contains Python scripts used to run queries and train models for this work.

## Set Up and Install

To run the code, first download this repository to your local machine. Navigate to the cloned folder on your local machine. Then, create a new Anaconda
environment with the `environment.yml` file:

```
conda env create -f environment.yml
```

Finally, install an editable version of this package using `pip`. Be sure to run
the following command in the top level folder, where `pyproject.toml` is visible:

```
pip install -e .
```

You should now have access to access the functions in this codebase as importable modules
anywhere in your virtual environment.