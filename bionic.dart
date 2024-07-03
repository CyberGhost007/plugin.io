import 'dart:math';

class ContentType {
  final String name;
  final double fixationRatio;
  final double opacity;
  final double emphasisFactor;

  const ContentType(
      this.name, this.fixationRatio, this.opacity, this.emphasisFactor);
}

class EnhancedAdaptiveBionicReader {
  static const Map<String, ContentType> predefinedContentTypes = {
    'default': ContentType('default', 0.5, 0.7, 1.0),
    'general': ContentType('general', 0.5, 0.7, 1.0),
    'technical': ContentType('technical', 0.6, 0.8, 1.2),
    'academic': ContentType('academic', 0.55, 0.75, 1.1),
    'narrative': ContentType('narrative', 0.45, 0.65, 0.9),
    'news': ContentType('news', 0.5, 0.7, 1.0),
    'casual': ContentType('casual', 0.4, 0.6, 0.8),
  };

  static final Set<String> _commonWords = {
    'the',
    'be',
    'to',
    'of',
    'and',
    'a',
    'in',
    'that',
    'have',
    'i',
    'it',
    'for',
    'not',
    'on',
    'with',
    'he',
    'as',
    'you',
    'do',
    'at',
    'this',
    'but',
    'his',
    'by',
    'from',
    'they',
    'we',
    'say',
    'her',
    'she',
    'or',
    'an',
    'will',
    'my',
    'one',
    'all',
    'would',
    'there',
    'their',
    'what'
  };

  static String process(
    String text, {
    String contentType = 'general',
    double? baseFixationRatio,
    double? baseOpacity,
    double readingSpeedFactor = 1.0,
    double? emphasisPreference,
  }) {
    ContentType selectedType = predefinedContentTypes[contentType] ??
        predefinedContentTypes['general']!;

    baseFixationRatio ??= selectedType.fixationRatio;
    baseOpacity ??= selectedType.opacity;
    emphasisPreference ??= selectedType.emphasisFactor;

    List<String> words = text.split(RegExp(r'\s+'));
    List<String> processedWords = [];

    for (int i = 0; i < words.length; i++) {
      double progressFactor = i / words.length;
      double adaptiveRatio = _getAdaptiveFixationRatio(
          words[i],
          i,
          words.length,
          baseFixationRatio,
          readingSpeedFactor,
          contentType,
          progressFactor);
      double adaptiveOpacity = _getAdaptiveOpacity(words[i], i, words.length,
          baseOpacity, emphasisPreference, contentType, progressFactor);
      processedWords
          .add(_createFixation(words[i], adaptiveRatio, adaptiveOpacity));
    }

    return processedWords.join(' ');
  }

  static String _createFixation(
      String word, double fixationRatio, double opacity) {
    RegExp wordPattern = RegExp(r'(\W*)(.*?)(\W*)$');
    Match? match = wordPattern.firstMatch(word);

    if (match == null) return word;

    String prefix = match.group(1) ?? '';
    String core = match.group(2) ?? '';
    String suffix = match.group(3) ?? '';

    if (core.length <= 3) {
      return '$prefix<span style="font-weight: bold; opacity: $opacity;">$core</span>$suffix';
    } else {
      int fixationLength = max(1, (core.length * fixationRatio).round());
      String fixation = core.substring(0, fixationLength);
      String remainder = core.substring(fixationLength);
      return '$prefix<span style="font-weight: bold; opacity: $opacity;">$fixation</span>$remainder$suffix';
    }
  }

  static bool _isCommonWord(String word) {
    return _commonWords.contains(word.toLowerCase());
  }

  static double _getAdaptiveFixationRatio(
      String word,
      int position,
      int totalWords,
      double baseRatio,
      double readingSpeedFactor,
      String contentType,
      double progressFactor) {
    double ratio = baseRatio;

    // Word length adjustment
    if (word.length > 8)
      ratio += 0.1;
    else if (word.length < 4) ratio -= 0.1;

    // Position adjustment
    if (position == 0 || position == totalWords - 1) ratio += 0.1;

    // Common word adjustment
    if (_isCommonWord(word)) ratio -= 0.1;

    // Reading speed adjustment
    ratio -= (readingSpeedFactor - 1) * 0.1;

    // Content type specific adjustment
    if (contentType == 'technical' || contentType == 'academic') {
      ratio += 0.05; // Increase fixation for technical/academic content
    } else if (contentType == 'casual' || contentType == 'narrative') {
      ratio -= 0.05; // Decrease fixation for casual/narrative content
    }

    // Progressive adaptation
    ratio -= progressFactor * 0.1;

    return ratio.clamp(0.2, 0.8);
  }

  static double _getAdaptiveOpacity(
      String word,
      int position,
      int totalWords,
      double baseOpacity,
      double emphasisPreference,
      String contentType,
      double progressFactor) {
    double opacity = baseOpacity;

    // Word length adjustment
    opacity += (word.length / 20) * emphasisPreference;

    // Position adjustment
    if (position == 0 || position == totalWords - 1)
      opacity += 0.1 * emphasisPreference;

    // Common word adjustment
    if (_isCommonWord(word)) opacity -= 0.1 * emphasisPreference;

    // Content type specific adjustment
    if (contentType == 'technical' || contentType == 'academic') {
      opacity += 0.05 *
          emphasisPreference; // Increase opacity for technical/academic content
    } else if (contentType == 'casual' || contentType == 'narrative') {
      opacity -= 0.05 *
          emphasisPreference; // Decrease opacity for casual/narrative content
    }

    // Progressive adaptation
    opacity -= progressFactor * 0.1 * emphasisPreference;

    return opacity.clamp(0.3, 1.0);
  }
}

void main() {
  String sampleText =
      "The quick brown fox jumps over the lazy dog. It was the best of times, it was the worst of times.";

  print("Original text:");
  print(sampleText);

  print("\nEnhanced Adaptive Bionic Reading (general content type):");
  print(
      EnhancedAdaptiveBionicReader.process(sampleText, contentType: 'general'));

  print("\nEnhanced Adaptive Bionic Reading (technical content type):");
  print(EnhancedAdaptiveBionicReader.process(sampleText,
      contentType: 'technical'));

  print("\nEnhanced Adaptive Bionic Reading (narrative content type):");
  print(EnhancedAdaptiveBionicReader.process(sampleText,
      contentType: 'narrative'));

  print(
      "\nEnhanced Adaptive Bionic Reading (custom settings with technical content type):");
  print(EnhancedAdaptiveBionicReader.process(sampleText,
      contentType: 'technical',
      readingSpeedFactor: 1.5,
      emphasisPreference: 1.3));
}
