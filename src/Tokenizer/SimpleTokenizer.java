package Tokenizer;

import java.io.*;
import java.util.ArrayList;

import org.wltea.analyzer.*;
import org.wltea.analyzer.core.*;

public class SimpleTokenizer
{
	public static void main(String[] args)
	{
		String text = "I'm on the roll, I'm on the roll this time.";
		String[] result = tokenizedText(text);
		for (String word : result)
			System.out.println(word);
	}
	
	static String[] tokenizedText(String text)
	{
		ArrayList<String> ret = new ArrayList<>();
		int index = 0;
		StringReader reader = new StringReader(text);
		IKSegmenter ik = new IKSegmenter(reader,true); 
	    Lexeme lexeme = null; 

	    try
		{
			while((lexeme = ik.next())!=null) 
				ret.add(lexeme.getLexemeText());
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	    
	    return (String [])ret.toArray(new String[ret.size()]);
	}
}
