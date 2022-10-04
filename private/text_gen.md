# Generation

Each framework has a generate method for auto-regressive text generation implemented in their respective `GenerationMixin` class:

- PyTorch [`~generation_utils.GenerationMixin.generate`] is implemented in [`~generation_utils.GenerationMixin`].
- TensorFlow [`~generation_tf_utils.TFGenerationMixin.generate`] is implemented in [`~generation_tf_utils.TFGenerationMixin`].
- Flax/JAX [`~generation_flax_utils.FlaxGenerationMixin.generate`] is implemented in [`~generation_flax_utils.FlaxGenerationMixin`].

## GenerationMixin


#### Generate

Text generation produces/creates a sequence of token ids for models with a language modeling head. It supports the following generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

- _`greedy decoding`_ by calling `greedy_search()` if `num_beams=1` and `do_sample=False`.
- _`multinomial sampling`_ by `calling sample()` if `num_beams=1` and `do_sample=True`.
- _`beam-search`_ decoding by `calling beam_search()` if `num_beams>1` and `do_sample=False`.
- _`beam-search multinomial sampling`_ by `calling beam_sample()` if `num_beams>1` and `do_sample=True`.
- _`diverse beam-search`_ decoding by `calling group_beam_search()`, if `num_beams>1` and `num_beam_groups>1`.
- _`constrained beam-search decoding`_ by calling `constrained_beam_search()`, if `constraints!=None` or `force_words_ids!=None`.


Apart from the `inputs`, all arguments below will default to the value of the attribute with the same name as defined in the model’s config `(config.json)` which in turn defaults to the [`PretrainedConfig`] of the model.


Most of these parameters are detailed more in [this blog post](https://huggingface.co/blog/how-to-generate).


Here are some examples:

#### Greedy Decoding:

``` python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Today I believe we can finally"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# generate up to 30 tokens
outputs = model.generate(input_ids, do_sample=False, max_length=30)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```
``` python
['Today I believe we can finally get to the point where we can make a difference in the lives of the people of the United States of America.\n']
```


#### Multinomial Sampling

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Today I believe we can finally"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# sample up to 30 tokens
torch.manual_seed(0)
outputs = model.generate(input_ids, do_sample=True, max_length=30)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```
``` python
['Today I believe we can finally get rid of discrimination," said Rep. Mark Pocan (D-Wis.).\n\n"Just look at the']
```

#### Beam-search decoding:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

sentence = "Paris is one of the densest populated areas in Europe."
input_ids = tokenizer(sentence, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```
```python
['Paris ist eines der dichtesten besiedelten Gebiete Europas.']
```
#### Greedy_search

The sequences of token ids for models are generated using the **greedy decoding** language modeling head. 

Below is an example:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# set pad_token_id to eos_token_id because GPT2 does not have a EOS token
model.config.pad_token_id = model.config.eos_token_id

input_prompt = "It might be possible to"
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

# instantiate logits processors
logits_processor = LogitsProcessorList(
    [
        MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id),
    ]
)
stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

outputs = model.greedy_search(
    input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
)

tokenizer.batch_decode(outputs, skip_special_tokens=True)
```
```python
["It might be possible to get a better understanding of the nature of the problem, but it's not"]
```
#### Sample


Samples utilise the **multinominal sampling** language modelling head to generate sequences of token ids. 

Here is an example:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# set pad_token_id to eos_token_id because GPT2 does not have a EOS token
model.config.pad_token_id = model.config.eos_token_id

input_prompt = "Today is a beautiful day, and"
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

# instantiate logits processors
logits_processor = LogitsProcessorList(
    [
        MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
    ]
)
# instantiate logits processors
logits_warper = LogitsProcessorList(
    [
        TopKLogitsWarper(50),
        TemperatureLogitsWarper(0.7),
    ]
)

stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

torch.manual_seed(0)
outputs = model.sample(
    input_ids,
    logits_processor=logits_processor,
    logits_warper=logits_warper,
    stopping_criteria=stopping_criteria,
)

tokenizer.batch_decode(outputs, skip_special_tokens=True)
```
```python
['Today is a beautiful day, and a wonderful day.\n\nI was lucky enough to meet the']
```
#### Beam_search

Sequences of token ids are generated for models with a language modeling head using **beam search decoding** .

Below is an example:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
)
import torch

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

encoder_input_str = "translate English to German: How old are you?"
encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


# lets run beam search using 3 beams
num_beams = 3
# define decoder start token ids
input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
input_ids = input_ids * model.config.decoder_start_token_id

# add encoder_outputs to model keyword arguments
model_kwargs = {
    "encoder_outputs": model.get_encoder()(
        encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    )
}

# instantiate beam scorer
beam_scorer = BeamSearchScorer(
    batch_size=1,
    num_beams=num_beams,
    device=model.device,
)

# instantiate logits processors
logits_processor = LogitsProcessorList(
    [
        MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ]
)

outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

```python
['Wie alt bist du?']
```

[[autodoc]] beam_sample


[[autodoc]] group_beam_search


[[autodoc]] constrained_beam_search

## TFGenerationMixin

[[autodoc]] generation_tf_utils.TFGenerationMixin
	- generate

## FlaxGenerationMixin

[[autodoc]] generation_flax_utils.FlaxGenerationMixin
	- generate
