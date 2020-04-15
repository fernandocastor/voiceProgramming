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

*Complication*: No dataset. Existing work on program synthesis and similar areas tries to generate code from high level descriptions. This is a harder problem, but it is made easier by the availability of datasets pairing method-level comments and the code itself. Even one-line functions suffer from that problem. Documentation is written at a high level of abstraction, in a way that accounts too much for the context where the function is used. This is probably very different from what a user would provide as input. 

*Complication*: Is this flavor of voice-based programming just natural language programming? I think not because existing approaches to natural language programming and program synthesis place a strong emphasis on abstract descriptions. Programming by voice is the opposite: the programmer wants finer grained control over what is generated. In addition, the kinds of errors that stem from voice recognition do not exist in plain natural language programming. 

- **Embrace English language speech recognizers** because they are trained much more intensively than a PL-based language model would ever be. This means that we should take (potentially incorrect) English text as a starting point. The downside to this is that our approach would need to be retailored for additional natural languages. 
- **Tackle this as an NLP problem**. In other words, pre-process the text to perform stemming, lemmatization, stop word removal. Also, look for typical homophones (again, "def" comes to mind)  and, based on context, remove them. The remark about context is necessary because, sometimes, a developer may really want to call a function "death".
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

Interesting website about data sets for source code-related ML:

https://github.com/src-d/awesome-machine-learning-on-source-code 

## Perturbations

Initially, we will introduce just the following types of perturbations:

- Removal of non-identifier token. How many? Can't be too many because the readings will tend to be short. 
- Introduction of spurious words. We cannot use WordNet for this. It's got too many words and also some multi-word expressions. It may be too much. Maybe get a corpus of common English words, such as this one: https://www.wordfrequency.info/free.asp. How many?
- Changes in the order. Cannot mix parameters and the function name. 
- Combinations of the above. 
  
### The actual sequence:

1. Collect a large number (100K+) method signatures. 
Maybe the corpus from this paper can help:
Antonio Valerio Miceli Barone, Rico Sennrich:
A Parallel Corpus of Python Functions and Documentation Strings for Automated Code Documentation and Code Generation. IJCNLP(2) 2017: 314-319

Complications: complex function names, complex parameter names, special parameters such as "*args" and "**kwargs", default parameter values, parameter and function names which as not comprised of actual words, e.g., `kwarg`. Most functions use "_" as the separator, but some of them also use camel case. Deal with camel case, we can probably use the Inflection library (https://pypi.org/project/inflection/), more specifically, the `underscore` function: https://inflection.readthedocs.io/en/latest/_modules/inflection.html#underscore.

Remove lines that start with the "@" symbol. Remove leading and trailing underscores.

Identifiers: 
- English words or
- Underscore-separated sequences of English words
- If a parameter comprises multiple words, there must be commas between them

1. Generate a bunch of method readings
2. Train a standard NN architecture for language translation

The labels do not use the actual function names. We removed the "_" characters because we are not doing character-based encoding. Since we're using word encoding, `a_function` would be interpreted as a new word, instead of a combination of `a` and `function`. For function declarations, it is easy to add 
the underscores later. That's not the case for other constructs. 

Forgot to remove parameter names including **. 

"build a function"


Natural language programming vs. Voice programming


paraphrases in French (for the kind of sentence we are using): often nonsensical. On the other hand, voice recognition often also produces nonsensical text, specially for stuff that is not in the language model.

Google imposes weird restrictions on the usage of its translation services. At the end: 6000+ translation requests for single sentences. For 0.5 per translation, that amounts to 50 minutes! Too long. Used a special sequence to break sentences so that I could translate larger character sequences. Using XXXXXXXXXXX does not work (becomes a weird word). Using %%%%%%%%%%%%%%%%%%% also does not work (gets changed in multiple ways). 

Expressions remain elusive


1. Evaluate the balance of the generated readings. Maybe rebalance based on that.
2. Create paraphrased versions of the methods using Spanish or French. Perform sanity checks on the paraphrased methods (to identify method and parameter names).
3. Augment the data with work omissions, comissions, and partial reorderings. 
4. Look for common misunderstandings (*"death"* for *"def"*, *"calling"* or *"colon"* for *":"*, *"funk"* for *"func"*, etc.)
5. Train a standard NN architecture for language translation
6.  Evaluate 

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

