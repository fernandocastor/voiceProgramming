# Recognizing Function Signatures

## Problems

The problems discussed here are general to programming by voice. However, other kinds of constructs, e.g., expressions and statements, include additional problemas that we do not discuss here. Problems to address:

- **Speech recognition is unreliable**. Current speech recognition technology, even when receiving training, may recognize key words, e.g., "function", "parameter", "plus", etc. In particular, *"def"* and a few others will need to be treated in a special way. Homophones matter for some particular words, such as "def". Another problem is single words that are recognized as multiple words and the inverse, but that's a very general problem of SR systems. 
- **Diversity of ways to express the same idea**. There is a wide variety of ways to tell the computer to do the same thing. How to account for that diversity of ways to express ideas in the training data?
- **Input may be partial**. Not only an idea can be expressed in different ways, but it can also be expressed partially. For example, a developer may first say "def head", pause, do somethig else, and then come back and say something like "with parameter l". 
- **Dealing with identifiers**. This is not a completely separate problem (it is also about the unreliability of speech recognition). However, it is harder than recognition a syntatic structure. Since identifiers may be a sequence of English words, abbreviations, mnemonics, or combinations of these elements, they will often be unknown words to the speech recognition system (also known as Out Of Vocabulary words). The vocabulary in a software development project tends to be richer than everyday natural language [1]. In addition, the problem of homophones is even stronger here due to the lack of syntactic context. We may have to constrain users of the system to deal with this.
- **Lack of training data**. Deep learning approaches have proven effective in language translation tasks. However, they rely on large amounts of labeled that for training. This data is not available for programming by voice because it is a niche activity performed in specific ways based on existing speech recognition systems. 

[1] A. Hindle, E. T. Barr, M. Gabel, Z. Su, and P. Devanbu. On the naturalness of software. Comm. ACM 59(5), pp. 122–131, May 2016, DOI:10.1145/2902362.

## Tackling the problems

Since our ultimate goal is to support programming by voice, a reasonable question to ask is whether we should go straight to processing voice input, instead of the text that is recognized by a speech recognition system. A good argument in favor of this is that existing deep learning models are very good at recognizing voice input. In this scenario, we partially break the separation between the problems of speech recognition and natural language processing. Our arguments for not going straight to voice and working with text instead:

- Voice data is harder to process and transform directly. In particular, it is easier to extract information from text than from voice.
- Requires more diversity and more data. Dealing with variations in background noise, accent, pronunciation, style, is a tough problem. Speech recognition software already addresses this.
- We cannot use existing language models because we would not be working at the language level
- For text, we have the hope of finding or synthesizing preexisting training data. For voice, that data does not exist. Arguably, it is possible to synthesize examples for voice as well,but the potential diversity in this case is much lower because it is limited by the constraints of existing speech synthesis systems.

With this in mind, we address the aforementioned problems by leveraging three main ideas: 

- **Deep learning**, because it is good for translation and can account for the sequencing of the features. One important question: *word* vs *character* embeddings. Apparently, character embedding may be a better option to tackle of problem of Out of Vocabulary words. Check this text:

  + https://towardsdatascience.com/character-level-cnn-with-keras-50391c3adf33

- **Embrace English language speech recognizers** because they are trained much more intensively than a PL-based language model would ever be. This means that we should take (potentially incorrect) English text as a starting point. The downside to this is that our approach would need to be retailored for additional natural languages. 
- **Tackle this as an NLP problem**. In other words, pre-process the text to perform stemming, lemmatization, stop word removal, and mostly ignore the order of the terms. Also, look for typical homophones (again, "def" comes to mind)  and, based on context, remove them. The remark about context is necessary because, sometimes, a developer may really want to call a function "death".
- **Synthesize training data**. Use existing functions to generate different ways of vocalizing function declarations, taking measures based on existing work on dataset diversification and synthesis to improve diversity. Examples of such measures include word removal, exchange by homophone, include spurious word, and exchange by spurious word. In the future (when we consider other options), we may also want to include actions related to words that are important for other syntactic constructs (e.g., variable) to simulate mixing in a more natural way. It is also possible to mix words on a character by character basis, but that only makes sense if we use characters as features, instead of words.

Check these texts:

  + https://towardsdatascience.com/synthetic-data-generation-a-must-have-skill-for-new-data-scientists-915896c0c1ae

  + http://openaccess.thecvf.com/content_cvpr_2016/papers/Gupta_Synthetic_Data_for_CVPR_2016_paper.pdf

  + https://www.robots.ox.ac.uk/~vgg/publications/2016/Jaderberg16/jaderberg16.pdf


## About synthesizing

Types of constructs, the minimal set to show that this idea is viable: 

- Highly structured constructs, with clear syntactic cues and relatively little variation, such as a function declaration
- Highly unstructured, highly varied in both nature, number, and order, and inherently recursive, such as expressions
- Is there a middle ground? Where do statements fit?

Lemmatize. Perform stemming. Remove stopwords. Maybe some other words as well.
Type the correct one, but also use multiple speech recognition libraries to get incorrect text. Generalize this.
Multiple variations, of course (including literal ones and accounting for typical errors)
Multiple functions
Enrichment:

- Omissions.
- Spurious words.
- Synonyms and maybe more specific terms (WordNet. What else?)
- We want training to account for order. 
- Combinations of these factors.
- Reading a random sample to different speech recognition systems (not necessarily multiple functions) to capture common speech recognition errors and include them in the model. I think the greatest problem, however, will be the identifiers and not so much the fixed parts, with obvious exceptions ("def", "func"). There are too many examples to read manually, however. Get these deviations and introduce them in the generation process. Identifiers are the worst obstacle, however.

Do we need incorrect words, except when incorrectly recognized? Does it make sense to employ a Mix strategy as is usually employed to enrich image datasets? For our case, that would be something like mixing two different kids of lines/declarations/statements.

Default argument values?

(wrong recognitions as well)
There is also ambiguity ("death" may be an identifier)
`abs(x)`
                                   def                                          id    ( par-list):
(**preamble**|**function-word** **named**?|(**preamble** **function-word**)) **name** **parameters**?

Additional rules (what kind of grammar is this? Context-dependent?):

- if preamble/function-word, nothing needs to be said about parameters (if the function does not take any). This is a context dependence. It's like variable declaration. 
- The optional "a" in the terminals for **function-word** should only appear if preamble appears. This is a context dependence. It's like variable declaration. 
- Most of the "function" synonyms come from WordNet.
- There are many more additional rules that are related to language and to how many readings for a single function one wants to generate. We're talking about potentially hundreds or even more than a thousand readings per function. **How to generalize?**

**preamble** = define | create | declare | write 
**function-word** = def | a? function | a? func | a? routine | a? subroutine | a? program | a? method | a? procedure
**named** = named | with name | called | whose name is
**name** = *an identifier*
**parameters** = **parenthesis**? **the-parameters** **parenthesis**?
**parenthesis** = between parenthesis | between parentheses | left parenthesis | ( | ) | right parenthesis | parenthesis | open parenthesis | close parenthesis
**the-parameter** = **par-prefix**? **pars** **par-suffix**
**par-prefix** = **par-prefix1** **par-prefix2**? 
**par-prefix1** = with | getting | receiving | taking 
**par-prefix2** = parameter | parameters | *number of parameters* parameters 
**pars** = (*an identifier* ,?)* (and *an identifier*)? 
**par-suffix** = (as parameter | as parameters | as argument | as arguments) :? (new line)?
 

About stop word removal for English:
https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

## Perturbations

Initially, we will introduce just the following types of perturbations:

- Removal of non-identifier token. How many? Can't be too many because the readings will tend to be short. 
- Introduction of spurious words. We cannot use WordNet for this. It's got too many words and also some multi-word expressions. It may be too much. Maybe get a corpus of common English words, such as this one: https://www.wordfrequency.info/free.asp. How many?
- Changes in the order. Cannot mix parameters and the function name. 
- Combinations of the above. 
  


## Unaddressed known limitations

No type annotations, no default values, and no unlimited parameters lists, at least for now. We ignore trailing and leading underscores. We use underscores to break down names. We do not account for underscores mentioned during voice programming, e.g., we do not, for now, consider that a function named "the_function" will be uttered "the", "underscore", "function". 
Our approach to break down function name identifiers does not work for parameter names because the parameters may be listed just based on their names, with no separation. Therefore, it would not be possible to determine if the result of removing the underscore from  "format_string" is a single parameter with a composite name or two parameters. 

Dealing with *camelCase* or *separated_by_underscores* identifiers.

GENERALIZING IS HARD!!!

Perturbations (changing order)
Multiple senteneces when uttering declaration.

"the partial utterance problem" => For now, no breaks.

Identifiers with just one word or single letter plus numbers 

Paraphrase generation. => We are not dealing exactly with paraphrasing from a linguistic because there are multiple details that can vary, e.g., just saying the names of the parameters instead of explicitly separating them with commas, but the problem is very similar. 
One problem is the same one that motivated us to generate readings in the first place: there is no dataset for training. 

Obvious ambiguity due to speech recognition:
',' => coma, comma
'def' => death, deaf
'func' => funk
 'and' => end (when listing items, such as a parameter list.)
 'colon' => 'column', 'calling'
Not tackling partial readings. Must also be generated, broken down among the parts of a definition.

This text is very interesting: https://medium.com/@wolfgarbe/the-average-word-length-in-english-language-is-4-7-35750344870f
It talks about typical word lengths, edit distances (for typical errors, among other things). Even more interesting, by following it cursorily, I got to a database of typical mispellings in English: https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/0643/allzip. Don't know if it will be useful, but it may be. 

