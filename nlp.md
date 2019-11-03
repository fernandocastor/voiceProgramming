# Natural Language Processing

## 1. Language Processing

Processing natural language text requires us to address a number of complications that do not arise when handling programming language text. For example, it is necessary to perform **Word Sense Disambiguation** because the same word can have different meanings depending on the **context** in which they appear. An example is the word *"by"*, which may have an **agentive** meaning, e.g., "The refugees were relocated *by* the government", a **temporal** meaning, e.g., "The refugees were relocated *by* night", or a **locative** meaning, e.g., "The refugees were relocated by sea". For each of these examples, the actual meaning of "by" is given by the expression following it. The last one actually has a peculiarity: if it were "by the sea", the meaning would be agentive instead of locative.

**Pronoun resolution** also poses a problem. This is not hard for simple sentences, but can become difficult for sequences of sentences where the pronoun of a sentence may refer to either subject or object of the previous one. The following example, extracted from NLTK, illustrates this very well: 

- The thieves stole the painting. They were subsequently caught.
- The thieves stole the painting. They were subsequently sold.
- The thieves stole the painting. They were subsequently found.

In the first case, "they" refers to the thieves, in the second, to the painting, and, in the third, it  may be either. Resolving pronouns requires techniques to, e.g., determine what are the subjects and the objects of the sentences. 

If these kinds of problems have been worked out, it is possible to **Generate Language Output**. Examples of the kinds of outputs that can be generated include answers to questions about the text or translations to other languages. Disambiguating the senses of words, resolving pronouns, and identifying subjects and objects of sentences are all steps in this direction.

**Textual entailment**: finding evidence to support a hypothesis in a short piece of text or stating that such evidence is not available. 


## 2. Accessing Text Corpora and Lexical Resources

**Lexical resources** enrich lists of words or sentences with additional information. Examples of lexical resources include pronunciation dictionaries that can be used for voice synthesis, comparative word lists that include how the words in the list would be written in multiple languages, and lists of synonyms for words. The latter provides very interesting information for us. For example, there are multiple sets of synonyms associated with the word "function", two of them listed below (extracted from WordNet):

- function mathematical_function single-valued_function map mapping
- routine subroutine subprogram procedure function

It is clear that the first one pertains to math whereas the second one pertains to programming. Also, it is notable that, among the programming synonyms of "function", the word "method" is not included. WordNet seems to be a useful lexical resource to help us parse NL sentences meant to be made into PL code. Besides synonyms, WordNet has a concept hierarchy. This allows us to delve into more specific (hyponyms) or general (hypernyms) concepts that are **lexically related** to another concept. For example, the concept of function in programming has the following hyponyms, according to WordNet: cataloged_procedure, contingency_procedure, library_routine, random_number_generator,recursive_routine, reusable_routine, supervisory_routine, tracing_routine,utility_routine, each one with its own set of synonyms. 

Unrelated: the **polysemy** of a word is the number of meanings it has.

One of the many interesting lexical resources in NLTK is the CMU Dictionary, which provides pronunciations for thousands of words. Could not find out how to load it from nltk, but it is possible to install it using pip, with package ``cmudict``.


## 3. Processing Raw Text

The ``word_tokenize`` function from ``nltk`` can perform tokenization. Lists of tokens can be used to create ``nltk.Text`` objects, which provides us with utilities to deal with text such as the ``concordance``function. 

This is not related, but it is very cool that Python has built-in libraries for reading (``feedparser`` library) blog feeds and for downloading the contents from web pages (``urllib`` library) very straightforwardly. Although not ideal, it is possible to read the contents of a web page with a one-liner (``raw = request.urlopen("THE://URL").read.decode("utf8")``). Processing the HTML requires a pip-accessible library (``bs4``) and can be made in the same line as the reading of the web page (by creating a ``BeautifulSoup`` object and invoking the ``get_text()`` method).

Another non-related topic that the text covers: regular expressions (module ``re``). For example, ``re.search("^..j..t..$", w)`` matches word ``w`` if it has exactly 8 characters that have 'j' as the third letter and 't' as the sixth one. On the other hand, ``re.search('^e-?mail$', w)`` matches both "email" and "e-mail", where character '-' is optional. It is possible to specify specific sets of characteres that will match. For example, the regular expression in ``[w for w in tokens if re.search('[xyz]..[abc]', w)]`` will match any word from ``tokens`` that has at least four characters where, in that four-character sequence, the first one is 'x', 'y', or 'z' and the last one is 'a', 'b', or 'c'. The ``-`` symbol denotes a range of characters, e.g., ``[w for w in tokens if re.search('[h-k][x-z]', w)]``  matches every word that includes a sequence of two adjacent characters, the first one in the range between 'h' and 'k' and the second in the range between 'x' and 'z'. The ``+`` symbol is, as expected, one of more, whereas ``*`` means zero or more. The character ``^``, when appearing within square brackets, acts as a negation, e.g., ``[^aeiou]`` matches anything except for lowercase vowels. About ``{}`` and ``|``, this example matches anything that has either at least two consecutive 'r' or two 'e': ``[w for w in tokens if re.search('(r{2,}|e{2,})', w)]``. The upper bound on the number of consecutive repetitions is left unspecified. 

Useful pattern for using regular expressions: ``re.findall(r'[aeiou]{2,}', raw)``. This finds all the sequences of characters (in this case, sequences of vowels of length 2 or more) in ``raw`` and returns them as a list. A quick note about parentheses: each part bound by parentheses becomes a member of a tuple and each member of this tuple corresponds to a matching element of the regular expression: 

```
>>> re.findall(r'^(p|q)(.*)(ing|ly|ed|ious)$', 'processing')
[('p', 'rocess', 'ing')]
```

In the example above, since none of the three parenthesized parts of the regular expression is optional, if any of them does not match, this expression will produce an empty list as its result.

The ``*`` operator attempts to match the longest possible string when it appears. In the following example, it will match "processe" and the second parenthesized part of the expression will match only "s":

```
>>> re.findall(r'^(.*)(s|es|es)$', 'processes')
[('processe', 's')]
```

From what I could grasp, it is possible to reduce its matching power so that it matches whatever remains after the non-optional parts have been matched, by using ``*?`` instead. In this case, the result changes and this does not stem from the ordering of the items in ``(s|es|ses)``: 

```
>>> re.findall(r'^(.*?)(s|es|ses)$', 'processes')
[('proces', 'ses')]
```

When processing natural language text, we usually want to normalize the text prior to processing. This normalization includes various activities such as making everything lower case and performing **stemming**. Stemming is the process of extraction the root part ("stem") of a word, that part which does not change depending on the word's context of use. The nltk includes multiple stemmers. Documentation recommends the use of the Porter stemmer, although it also mentions an alternative one named Lancaster. I've checked both and it is not obvious which one is the best. For Porter, "body" becomes "bodili", "wake" remains "wake" (what about "waking?"), and "come" remains "come" (what about "coming"?). Lancaster works more intuitively in these cases. However, besides failing with "lying", it sometimes seems to stem too much, e.g., "terror" becomes "ter", and "natural" becomes "nat", instead of "natur". There's a third option of stemmer, Snowball, which is a lot like Porter but is more intuitive to me. Whereas Porter turns "fairly" into "fairli", Snowball makes it into "fair". It is very easy to use nltk stemmers (assuming that raw stores a string): 

```
>>> porter = nltk.porter.PorterStemmer()
>>> stemmed = [porter.stem(w) for w in word_tokenize(raw)]
...
>>> snow = nltk.stem.SnowballStemmer('english')
>>> alsoStemmed = [snow.stem(w) for w in word_tokenize(raw)]  
```

Another form of normalization is **lemmatization**. Whereas stemming attempts to normalize words by removing suffixes that denote variant forms, usually based on simple patterns that are identified algorithmically, lemmatization typically uses a vocabulary and considers the context of use of the word (morphological analysis) in order to get its dictionary form. A better definition:

> **Stemming vs. Lemmatization**
From: http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form. However, the two words differ in their flavor. Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma. If confronted with the token "saw", stemming might return just "s", whereas lemmatization would attempt to return either "see" or "saw" depending on whether the use of the token was as a verb or a noun.

Nltk includes a lemmatizer based on WordNet. It is straightforward to use:

```
>>>lemma = nltk.wordnet.WordNetLemmatizer()
>>> lemma.lemmatize("churches)
church
```

Complementarily to the use of lemmatizers and stemmers, it is possible to use a dictionary such as Hunspell to suggest similar words. Installing Hunspell in Python 3.X requires two steps: 

1. ``brew install hunspell``
2. ``pip3.7 install cyhunspell``

If the steps above do not work, install GCC 6 (``brew install gcc@6``) prior to step 2. The main use I can see for Hunspell in this context is suggesting similar words in case a word is not what we expected:

```
>>> import hunspell
>>> h = hunspell.Hunspell()
>>> h.suggest("deaf")
('dead', 'def', 'decaf', 'dean', 'dear', 'deal', 'leaf')
```