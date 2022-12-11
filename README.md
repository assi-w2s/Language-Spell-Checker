# Spell-Checker
A noisy-channel based NLP Spell Checker.

This is a noisy-channel based spell checker, which is a type of spelling correction model that uses the noisy channel model to identify and correct spelling errors in natural language text. The noisy channel model is a well-known approach in natural language processing (NLP) that assumes that the observed text is the result of a noisy channel, where the original intended message is distorted by various sources of noise, such as spelling errors, typos, and other errors.

In the context of a spell checker, a noisy channel-based approach would involve using NLP techniques to identify and correct spelling errors in the observed text. In this model we're using keyboard spelling errors' data (errors.txt) to  model the likelihood of different spelling errors given the original intended message. In addition, I add context probabilities to the model, which refers to the use of context information to improve the performance of the spelling correction model (context can refer to the surrounding words and sentences in my model, but as well can include theoretically additional information, such as the topic, genre, or style of the text).

In general, this noisy channel-based spell checker can be used to identify and correct spelling errors in natural language text. This approach can provide more accurate and effective spelling correction than traditional spell checkers that rely on simple rule-based or dictionary-based approaches.
