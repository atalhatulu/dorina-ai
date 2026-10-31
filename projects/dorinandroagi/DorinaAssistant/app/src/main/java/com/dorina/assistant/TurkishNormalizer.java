package com.dorina.assistant;

import java.util.Locale;

public class TurkishNormalizer {

    public static String normalize(String text) {
        if (text == null) return "";
        
        // Convert to lowercase with Turkish locale first
        String lower = text.toLowerCase(new Locale("tr", "TR")).trim();
        
        // Replace Turkish specific chars and common misspellings/transliterations
        StringBuilder sb = new StringBuilder();
        for (char c : lower.toCharArray()) {
            switch (c) {
                case 'ı':
                case 'İ':
                case 'i':
                    sb.append('i');
                    break;
                case 'ş':
                    sb.append('s');
                    break;
                case 'ğ':
                    sb.append('g');
                    break;
                case 'ü':
                    sb.append('u');
                    break;
                case 'ö':
                    sb.append('o');
                    break;
                case 'ç':
                    sb.append('c');
                    break;
                default:
                    sb.append(c);
                    break;
            }
        }
        return sb.toString();
    }
}
