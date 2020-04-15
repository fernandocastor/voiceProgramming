# Code Completion From Abbreviated Input by Sangmok Han, David R. Wallace, and Robert C. Miller 

This paper presents an approach to building autocompletion engines that (i) can autocomplete multiple words in a statement or expression, e.g., ``str nm = th.gv(r,c)`` can become ``String name = this.getValueAt(row, col)``and (ii) does not require that abbreviated keywords be established in advance. To achieve this, it leverages an extended Hidden Markov Model where the emission probability is replaced by a match probability. The emission probability is the probability of a hidden state (an abbreviation, e.g., ``str``) generating an output symbol (a complete word, e.g., ``String``). This probability cannot be used because since the abbreviations are not preestablished, there are infinitely many possibilities. This probability is replaced by the match probability, which is either 0 or 1 and determined by a previously analyzed corpus of programs.

This is an interesting approach. I think it is limited in that it must match abbreviations to actual words one by one. This seems to be a common issue in this kind of approach. Keyword programming does not suffer from this, but it suffers from other issues, e.g., scalability (if there are multiple levels of indirection), imprecision.