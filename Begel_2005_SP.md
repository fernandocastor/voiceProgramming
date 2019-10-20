# Spoken Programs, by Andrew Begel and Susan Graham

This paper tackles the problem of supporting programs in writing programs in a typical programming language, for this specific case, Java, by using their voice. In other words, program code is not typed, but uttered. The paper says in its first paragraph that a *spoken command language* is a different* problem that is not addressed in this work. By "spoken command language", do they mean the navigation-related part of the voice input?

Interesting question: What would be a natural way to speak a programming language that also has a tractable, comprehensible, and predictable mapping to the original language? Complementarily, this work is *not* about programming using natural language. Instead, it is about how spoken natural language can be processed so as to build programs in a general purpose programming language. 

Four categories of differences between written and spoken code, when considering the input: lexical, syntactic, semantic, prosodic. The latter refers to the rhythmic and intonational aspect of the language. Prosody is often used to disambiguate words in spoken English, according to Begel and Graham. The differences is semantic exist, of course, but it is not clear to me how they impact the problem of receiving input. 

Clear message: ways to deal with ambiguity must be built into the programming language and possibly its accompanying environment if it aims to deal with voice input. The input has no punctuation, some words sound very similar, the structure of natural language differs greatly from the structure of a programming language. 

The preliminary study about how programmers read code involved only ten graduate students (who the paper called "expert programmers"). They were asked to read a single page of Java code. Five of them were not familiar to Java. The participants were asked to read the code as if they were telling a second year undegraduate student what to type. This latter requirement helps explain why the authors claim that the ten participants verbalized the code more of less in the same way. There were noteworthy differences, however. For example, ``array[i]`` could have been verbalized as "array sub i", "array of i", or "i from array". In addition, there is ambiguity, for example, "for" can be recognized as "4", "four", "fore", or "for". All of them sound the same. In a similar vein, "less than" can be literally "less than" or "<". Cases where camel case is desired can also create problems. 

Another complication, quoted literally: *"There are many stop words, false starts, restated expressions and statements, and stream of consciousness utterances sprinkled throughout spoken code"*. 

A useful example: how would we say f().g()? 

(for more literal to more abstract)
- "f left parenthesis right parenthesis dot g left parenthesis right parenthesis"
- "f parentheses dot g parentheses" (ambiguous)
- "call f no parameters dot call g no parameters"
- "call f dot call g"
- "call g on the result of calling f"

Another complex example: array[i++] vs. array[i]++. There is more than one way of saying this and there is ambiguity. Type information can help reduce this ambiguity, if it is available. In Begel and Graham's study, English speakers inserted a pause as a means to separate parts that should not be grouped together. Non-native English speakers adopted a different, semantic strategy, e.g., "increment the ith element of the array" (for the second example above).  

I am starting to think that statements and expressions have to be more concretely spoken than declarations (this is not in the paper). There may be shortcuts for some expressions (such as function calls, constructor calls, and for loops). Declarations, differently, can be built as templates since there are much fewer.

About punctuation, it has to be verbalizable and work in both context-sensitive and a context-insensitive ways.

Another thing that is not in the paper: it would be really nice to support profiles for different languages within the same intermediate representation, in particular, Portuguese and English.

According to the authors, the grammar of SpokenJava includes only 11 additional rules that address three scenarios: (i) the opening and closing curly braces in class and interface declarations; (ii) different forms of verbalizing empty argument lists and how to differentiate them from lists with at least one argument; and (iii) an alternate phrasing for assignment (set x to a). I think there is a lot more room for ambiguity in the grammar than these three scenarios. 

In English, recognizing single letters without using  military/aviation alphabet seems to be hard. Entering complex mathematical expressions is also a difficult problem. Ref. [7] tackles this problem. 

References [13] and [14] propose something called NaturalJava. Uses a specially developed natural language. References [8] and [18] deal with the output aspect of programming by voice.

One open question (in 2015) supporting programmers in inputting incomplete programs. In addition, this work has focused on how students vocalize code but not what kinds of errors they make in that process. This would be useful for a production system. 

There is also voice strain. Can that requirement be accounted for by us, for example, by incentivizing pauses?