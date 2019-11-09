# SpeechClipse – An Eclipse Speech Plug-in, by Shairaj Shaik and colleagues

**Speech Technology**: Speech Synthesis and Speech Recognition

An "adaptive" speech plugin for Eclipse. It is based on the Java speech API. No implementation of this API ships with the JDK. Third parties implement it. This is a very old API and even the Oracle webpage for the JSAPI (https://www.oracle.com/technetwork/java/jsapifaq-135248.html) makes multiple references to Sun Microsystems.

They used something called the Java Speech Grammar Format to specify the grammar for SpeechClipse. More specifically, it used a "rule-based" grammar to recognize commands. The authors mention that a "dictation-based" grammar would have been more flexible and appropriate to support the goal of unrestricted natural language input. 

From what I could grasp, the focus here is on navigation and on operating Eclipse by voice. It supports code input in a limited form. According to the text of the paper: 

> *"SpeechClipse currently
supports a basic form of spoken code writing
facility called LazyTyping. Users can dictate
well known programming language keywords,
but more work is needed in making LazyTyping
flexible."*

Raises the point of homophones ("for" vs. "fore") and how to display spoken numbers ("1" vs. "one").