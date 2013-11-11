package Tokenizer;

import java.io.IOException; 
import java.io.StringReader; 

import org.wltea.analyzer.*;
import org.wltea.analyzer.core.*;

/** 
* Hello world! 
* 
*/ 

public class IKAnalyzerDemo 
{ 
    public static void main( String[] args ) throws IOException{ 
    String str = "today is a nice day, Guangzhou EverGxxxx wins the cup."; 
    StringReader reader = new StringReader(str); 
    IKSegmenter ik = new IKSegmenter(reader,true);//当为true时，分词器进行最大词长切分 
    Lexeme lexeme = null; 
    while((lexeme = ik.next())!=null) 
    System.out.println(lexeme.getLexemeText()); 
    } 
} 