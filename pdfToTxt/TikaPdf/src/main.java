import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.pdf.PDFParser;
import org.apache.tika.sax.BodyContentHandler;

import org.xml.sax.SAXException;

public class main {
	
   public static void main(final String[] args) throws IOException,TikaException, SAXException {

      BodyContentHandler handler = new BodyContentHandler(-1);
      Metadata metadata = new Metadata();
      //get current directory
      Path currentPath = Paths.get(System.getProperty("user.dir"));
      //set up input and output directory path
      Path inputPath = Paths.get(currentPath.toString(), "input");
      Path outputPath = Paths.get(currentPath.toString(), "output");
      File folder = inputPath.toFile();
      File[] listOfFiles = folder.listFiles();
      
      for(int i = 0; i < listOfFiles.length; i++)
      {
    	  String fileName = listOfFiles[i].getName();
    	  if(listOfFiles[i].isFile()&& !(fileName.charAt(0) == '.'))
    	  {
    		  System.out.println("working on file:" +listOfFiles[i]);
    		  //String fileName = listOfFiles[i].getName();
    		  fileName = fileName.replaceAll(".pdf", ".txt");
    		  FileInputStream inputstream = new FileInputStream(listOfFiles[i]);
    		  ParseContext pcontext = new ParseContext();
    		  
    		  //parsing the document using PDF parser
    		  PDFParser pdfparser = new PDFParser(); 
    	      pdfparser.parse(inputstream, handler, metadata,pcontext);
    	      
    	    //getting the content of the document
    	      //System.out.println("Contents of the PDF :" + handler.toString());
    	      Path outputFilePath = Paths.get(outputPath.toString(), fileName);
    	      PrintWriter writer = new PrintWriter(outputFilePath.toString());
    	      writer.write(handler.toString());
    	      writer.close();
    	      
    	      System.out.println(listOfFiles[i]+" Completed");
    	  }
    	  else
    	  {
    		  System.out.println(listOfFiles[i]+ "is not a file.");
    	  }
      }
      
      /*
      FileInputStream inputstream = new FileInputStream(new File("test.pdf"));
      ParseContext pcontext = new ParseContext();
      
      //parsing the document using PDF parser
      PDFParser pdfparser = new PDFParser(); 
      pdfparser.parse(inputstream, handler, metadata,pcontext);
      
      //getting the content of the document
      System.out.println("Contents of the PDF :" + handler.toString());
      
      //getting metadata of the document
      System.out.println("Metadata of the PDF:");
      String[] metadataNames = metadata.names();
      
      for(String name : metadataNames) {
         System.out.println(name+ " : " + metadata.get(name));
      }*/
   }
}