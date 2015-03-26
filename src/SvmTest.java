


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
		
		long startMili=System.currentTimeMillis();// 当前时间对应的毫秒数
		System.out.println("开始");
		
//		String inputPath = args[0];
//		String outputPath = args[1];
		String inputPath = ".";
		String outputPath = ".";
		
		svm_train.getPath(inputPath);
		
		/******参数寻优********/
		String[] para_trainArgs = {"-h","0","-v","3",inputPath+"/trainWordsVector.txt"};//directory of training file
		/******参数测试********/
//		String[] para_trainArgs = {"-h","0","-c",Double.toString(Math.pow(2, 11)),"-g",Double.toString(Math.pow(2, -17)),inputPath+"trainWordsVector.txt"};//directory of training file
		svm_train.main(para_trainArgs);

		/******最优参数model训练********/
		String[] trainArgs = {"-h","0","-c",Double.toString(Math.pow(2, svm_train.bestC)),"-g",Double.toString(Math.pow(2, svm_train.bestG)),
				inputPath+"/trainWordsVector.txt",outputPath+"/trainWordsVector.txt.model"};//directory of training file
		String modelFile = svm_train.main(trainArgs);
		
		
		/******准确率测试********/
		String[] testArgs = {inputPath+"/testWordsVector.txt", modelFile,outputPath+ "/resultWordsVector.txt"};//directory of test file, model file, result file
		Double accuracy = svm_predict.main(testArgs);
		System.out.println("SVM Classification is done! The accuracy is " + accuracy);
			

		/******生成对应范围的参数测试集********/
//		PrintWriter pw = new PrintWriter(new FileWriter(inputPath+"/Tmp/testParameters.txt"));
//		
//		for(double c =-1;c <= 14.0001;c++)
//			for (double g = -23;g <= -7.9999;g++)
//				pw.println("0.0  "+"1:"+c+" 2:"+g);
//		
//		pw.close();
		
		long endMili=System.currentTimeMillis();
		System.out.println("结束");
		System.out.println("总耗时为："+(endMili-startMili)/1000.0+"秒");
		
	}
}