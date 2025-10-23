Lab 3 Reflection

Preprocessing

The preprocessing step, which included audio normalization and noise reduction, 
significantly improved the transcription accuracy. Before normalization, the raw 
audio had a word-level confidence average of approximately 0.72, with some misrecognized 
words in the AI paragraph. After preprocessing, the average confidence increased to 0.94, 
and the transcription correctly captured my name and most of the AI paragraph. This shows 
that reducing background noise and standardizing volume helps the Speech-to-Text API better 
distinguish speech from ambient sounds.

PII Detection

Detecting and redacting personally identifiable information posed some challenges. 
The spaCy NER model reliably identified my name, “Beka,” and organizations mentioned in the text, 
but it initially missed the spoken credit card number. Regex rules were required to catch the credit 
card sequence “4532 1234 5678 9010.” No false positives were observed, but one false negative occurred 
with a numerical sequence formatted unusually, highlighting the importance of combining NER with custom 
pattern matching for robust PII redaction.

Confidence Scoring

Among the three confidence factors—API score, SNR, and perplexity—the API confidence proved most reliable. 
For my test audio, the API reported an average confidence of 0.91, while SNR was 38 dB and perplexity was 1.1. 
When combined using the weighted formula (0.6 API, 0.25 SNR, 0.15 perplexity), the resulting confidence score was 0.89,
labeled “HIGH.” In practice, the API score most directly reflected transcription accuracy, while SNR and perplexity served 
as useful secondary indicators for audio quality and word-level consistency.

Production Considerations

Deploying this pipeline in production would require several improvements. Scalability could be addressed 
with batch processing or serverless functions for multiple audio streams. Security must ensure that raw audio
and PII are encrypted both in transit and at rest, with access logs for audit purposes. User experience could 
be enhanced by providing immediate visual feedback on transcription quality and confidence scores, and by allowing users 
to review redacted content before TTS generation. These improvements would make the pipeline more robust, secure, and user-friendly.
