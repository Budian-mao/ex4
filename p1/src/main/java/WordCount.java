import java.io.IOException;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
public class WordCount {
    public static class DecreasingComparator extends Text.Comparator {
        @SuppressWarnings("rawtypes")
        public int compare(WritableComparable a, WritableComparable b){
            return -super.compare(a, b);
        }
        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            return -super.compare(b1, s1, l1, b2, s2, l2);
        }
    }

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable>
    {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        @Override
        protected void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException{
            String line=value.toString();
            String industry=line.split(",")[10];
            if(!industry.equals("industry")) {
                word.set(line.split(",")[10]);
                context.write(word, one);
            }
        }
    }

    public static class SortMapper extends Mapper<Object, Text, IntWritable, Text>{
        protected void map(Object key, Text value, Mapper<Object, Text, IntWritable, Text>.Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] keyValueStrings = line.split("\t");
            if (keyValueStrings.length != 2) {
                System.err.println("string format error!!!!!");
                return;
            }
            int outkey = Integer.parseInt(keyValueStrings[1]);
            String outvalue = keyValueStrings[0];
            context.write(new IntWritable(outkey), new Text(outvalue));
        }
    }

    public static class IntSumReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static class SortReducer extends Reducer<IntWritable, Text, Text, IntWritable>
    {
        protected void reduce(IntWritable key, Iterable<Text> values, Reducer<IntWritable, Text, Text, IntWritable>.Context context) throws IOException, InterruptedException {
            Iterator var = values.iterator();
            while(var.hasNext()) {
                Text value = (Text)var.next();
                context.write(value, key);
            }
        }
    }

    public static void main(String args[]) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf=new Configuration();
        Job job1=Job.getInstance(conf,"word count");
        job1.setJarByClass(WordCount.class);
        job1.setMapperClass(TokenizerMapper.class);
        job1.setCombinerClass(IntSumReducer.class);
        job1.setReducerClass(IntSumReducer.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);
        Path tmp=new Path("output");
        FileInputFormat.setInputPaths(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, tmp);
        if(job1.waitForCompletion(true))
        {
            Job sortjob=Job.getInstance(conf,"sort job");
            sortjob.setJarByClass(WordCount.class);
            sortjob.setMapperClass(SortMapper.class);
            sortjob.setReducerClass(SortReducer.class);
            sortjob.setMapOutputKeyClass(IntWritable.class);
            sortjob.setMapOutputValueClass(Text.class);
            sortjob.setSortComparatorClass(DecreasingComparator.class);
            FileInputFormat.setInputPaths(sortjob, tmp);
            FileOutputFormat.setOutputPath(sortjob, new Path(args[1]));
            System.exit(sortjob.waitForCompletion(true) ? 0 : 1);
        }
    }
}
