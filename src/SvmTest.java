


import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;



public class SvmTest extends Thread{

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		long startMili=System.currentTimeMillis();// ��ǰʱ���Ӧ�ĺ�����
		System.out.println("��ʼ");
		
//		String inputPath = args[0];
//		String outputPath = args[1];
		String inputPath = ".";
		String outputPath = ".";
		
		svm_train.getPath(inputPath);
		
		/******����Ѱ��********/
		String[] para_trainArgs = {"-h","0","-v","3",inputPath+"/trainWordsVector.txt"};//directory of training file
		/******��������********/
//		String[] para_trainArgs = {"-h","0","-c",Double.toString(Math.pow(2, 11)),"-g",Double.toString(Math.pow(2, -17)),inputPath+"trainWordsVector.txt"};//directory of training file
		svm_train.main(para_trainArgs);

		/******���Ų���modelѵ��********/
		String[] trainArgs = {"-h","0","-c",Double.toString(Math.pow(2, svm_train.bestC)),"-g",Double.toString(Math.pow(2, svm_train.bestG)),
				inputPath+"/trainWordsVector.txt",outputPath+"/trainWordsVector.txt.model"};//directory of training file
		String modelFile = svm_train.main(trainArgs);
		
		
		/******׼ȷ�ʲ���********/
		String[] testArgs = {inputPath+"/testWordsVector.txt", modelFile,outputPath+ "/resultWordsVector.txt"};//directory of test file, model file, result file
		Double accuracy = svm_predict.main(testArgs);
		System.out.println("SVM Classification is done! The accuracy is " + accuracy);
			

		/******���ɶ�Ӧ��Χ�Ĳ������Լ�********/
//		PrintWriter pw = new PrintWriter(new FileWriter(inputPath+"/Tmp/testParameters.txt"));
//		
//		for(double c =-1;c <= 14.0001;c++)
//			for (double g = -23;g <= -7.9999;g++)
//				pw.println("0.0  "+"1:"+c+" 2:"+g);
//		
//		pw.close();
		
		long endMili=System.currentTimeMillis();
		System.out.println("����");
		System.out.println("�ܺ�ʱΪ��"+(endMili-startMili)/1000.0+"��");
		
	}
}