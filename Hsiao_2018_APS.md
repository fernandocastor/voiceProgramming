# Automated Program Synthesis from Object-Oriented Natural Language for Computer Games by Michael S. Hsiao

This paper proposes a domain-specific approach to program using natural language in the domain of electronic games. The input uses what the author calls "object-oriented natural language" and every input NL command must involve an object, *" an identifiable entity
used in a video game, such as the specific characters, score, etc."* Apparently, this work does not worry much about ambiguity: *"If there are any ambiguity or unclear sentences, error messages and hints on how to fix the errors will be given as well."*

I think claiming that this is program synthesis sounds a big exaggerated. To me, it looks more like a translation, tailored to deal with the ambiguity of a constrained form o natural language and supported by a lot of preexisting code that is either already implemented in a library or is generated in a standard way. The abstract of the paper emphasizes that it is possible to generate 1800(!) lines of code based on only 20 instructions, but does not exactly show how this occurs.

It considers the input text as a set (order and repetitions do not matter). However, it later processes the text to identify synonyms (and replaces them so as to reduce vocabulary), tenses, modifier clauses, and pronouns. It also uses N-grams to identify nouns to which pronouns refer. It explicitly does not consider "the full grammatical structure of the sentence". Passive verbs are converted into active ones and (I am assuming) verb in different tenses are lemmatized. 

The proposed approach also uses N-grams to assign semantics. These n-grams associate actions, subjects, and objects. When this matters, order is taken into account, e.g., A eats B. When it does not matter, it is ignored, e.g., A and B collide. These N-grams are treated as predicates, e.g., eat(A,B). The grammar rules specifies how these actions over objects can occur.

The kind of ambiguity this approach deals with is different from what must be addressed in voice-based programming. For example, this is the example provided in the paper: *"A eats pushes B"*.

The system actually exists and is available at https://gc.ece.vt.edu/. The demo makes it clear that, as powerful as it may be, the proposed approach is domain-specific and tailored to a very specific kind of game. 

