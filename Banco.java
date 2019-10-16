
import java.io.File;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.trees.J48;


public class Banco {
	public static void main(String[] args) throws Exception {
		
		//Carrega o arquivo CSV
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("bank_classificador.csv"));
		Instances dado = loader.getDataSet();
		//Indica qual o atributo vamos classificar 
		dado.setClassIndex(dado.numAttributes()-1);
		
		//Carrega o modelo salvo
		J48 arvore = (J48) weka.core.SerializationHelper.read("modelo_j48.model");
		
		//Pega uma linha do arquivo e classifica
		double resultado = arvore.classifyInstance(dado.instance(0));
		System.out.println("resultado da classificação: " + resultado);
		
		
	}

	

}
