# METHOD AND SYSTEM FOR EFFICIENT VOICE-BASED PROGRAMMING, by Karl J. Weinmeister

This patent proposes a system to support voice-based programming. The main idea is to use the VoiceXML exchange format to match the results of speech recognition with the grammar of a programming language. The system is responsible for adding the punctuations symbols, parentheses, etc. The authors believe that the use of this exchange format can support voice-based programming for a number of different programming languages. The style of input is fairly literal and does not seem to account for deviations, imprecisions in speech recognition, homophones, etc. 

The text of the patent provides a single example of input and the recognized program:

> *"new static class factorial; main input limit; int fact; for i 1 limit; increment fact by limit; endparen; out(fact); endmain"*

According to the authors, for this (if correctly recognized) text, the recognized program would be:

```
public static class Factorial {
  public static void main (int limit) {
    int fact=0
    for (i-0; i-limit; i++) {
      fact+=Varnum;
      System.out.println(fact);
    }
  }
}
```

The patent does not make it clear whether the system was actually implemented. The patent was filed for a company called Nuance Communications, Inc. This is the company that develops **Dragon NaturallySpeaking**, one of the more well-known speech recognition systems and a pioneer in this area. According to the company's website:

> *"Nuance leads innovation in conversational AI, making intuitive, award-winning technology that adapts to each business and every unique situation. Our solutions don't just hear and speak. They understand, analyze, anticipate, reason and resolve. They don't just make life easier; they make it easy to achieve—and exceed—your goals."* 

