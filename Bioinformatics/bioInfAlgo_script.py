# -*- coding: utf-8 -*-
# Script for Lesson 1 from Bioinformatics Algorithms Part I



# PATTERN COUNT
#
def patternCount(text,pattern):
    '''
    Counts the number of occurences of pattern within text
    '''
    count = 0
    for J in range(len(text)-len(pattern)):
        if text[J:J+len(pattern)]==pattern:
            count = count+1
    return count 
#
# # READ INPUTS AND RUNS
# f = open('patternCount_text.txt', 'r')
# inputText = f.read()
# f.close()
# f = open('patternCount_pattern.txt', 'r')
# inputPattern = f.read()
# f.close()
# 
# print patternCount(inputText,inputPattern)



# FREQUENT WORDS
#
def frequentWords(text,k):
    '''
    Finds the pattern(s) of lenght k that appear the largest number of times
    '''
    frequentPatterns = ()
    count = [0]*(len(text)-k)
    for J in range(len(text)-k):
        pattern = text[J:J+k]
        count[J] = patternCount(text,pattern)
    maxCount = max(count)
    for J in range(len(text)-k):
        if count[J]==maxCount:
            frequentPatterns = frequentPatterns+(text[J:J+k],)
    return set(frequentPatterns)
#
# # READ INPUTS AND RUNS
# f = open('frequentWords_text.txt', 'r')
# inputText = f.read()
# f.close()
#
# print frequentWords(inputText, 14)



# REVERSE COMPLEMENT
def reverseComplement(text):
    '''
    Produces the reverse complement of any given text
    '''
    revCompText = ''
    for J in range(len(text)-1,-1,-1):
        x = text[J]
        if x=='A':
            revCompText += 'T'
        if x=='T':
            revCompText += 'A'
        if x=='C':
            revCompText += 'G'
        if x=='G':
            revCompText += 'C'
    return revCompText                    
#                     
# READ INPUTS AND RUNS
# f = open('reverseComplementExample_text.txt', 'r')
# inputText = f.read()
# f.close()
#
# print reverseComplement(inputText)



# PATTERN MATCHING
def patternMatching(pattern,genome):
    '''
    Finds all the starting points of pattern within genome
    '''
    startings = ""
    lenPattern = len(pattern)
    for J in range(len(genome)-len(pattern)):
        if genome[J:J+lenPattern]==pattern:
            startings = startings+' '+str(J)
                
    return startings     
#
# # READ INPUTS AND RUNS
# f = open('patternMatching_pattern.txt', 'r')
# pattern = f.read()
# f.close()
# f = open('patternMatching_genome.txt', 'r')
# genome = f.read()
# f.close()
#
# print patternMatching(pattern,genome)

 