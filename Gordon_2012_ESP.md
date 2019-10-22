# English for Spoken Programming, by Benjamin M. Gordon and George F. Luger

This is a longer version of "Developing a Language for Spoken Programming". All the comments there apply here. This time the authors actually present the proposed language. It is an imperative, C-like programming language whose main selling point is that most of the punctuation has been replaced by English words. The language is statically typed and leverages type inference so that type annotations are not required. The syntax for the language was derived from an "informal survey". Similarly to the "Spoken Programs" paper, does not support navigation. The code itself is very C-like, with a hello world program being something like:

```
define function main
taking no arguments as
  print the string "Hello, World"
  return 0
end function
```

An interesting aspect of this work is how it leverages scope to assist in speech recognition. Since the underlying speech recognition system, CMU Sphinx, does not allows for the modification of a language model at runtime, the system keeps one language model per scope so that identifiers that are closer to a given scope are more likely to be recognized. 

In addition, when faced with ambiguity in the names of identifiers, the tool asks Sphinx for potential candidates. From what I could grasp, the tool than attempts to determine the types of these candidates and match them with the expected type of the expression currently being input. This matching is employed to narrow down the number of possibilities in these scenarios. I think this is simplified by the language being statically-typed and by the impossibility of defining new types.   