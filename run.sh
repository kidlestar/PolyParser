#!/bin/bash

for value in bg ca col cs de en es fr it nl no ptb ro ru
do
   for i in {1..2}
   do
      name="exp$i"
      j=$((i+1))
      name1="exp$j"
      echo $name
      echo $name1
      sed -i "s/$name/$name1/g" n$value.slurm
      sbatch n$value.slurm
   done
done

for value in bg ca col cs de en es fr it nl no ptb ro ru
do
   name="exp3"
   name1="exp1"
   echo $name
   echo $name1
   sed -i "s/$name/$name1/g" n$value.slurm
   sbatch n$value.slurm
done

for value in bg ca col cs de en es fr it nl no ptb ro ru
do
   for i in {1..2}
   do
      name="exp$i"
      j=$((i+1))
      name1="exp$j"
      echo $name
      echo $name1
      sed -i "s/$name/$name1/g" $value.slurm
      sbatch $value.slurm
   done
done

for value in bg ca col cs de en es fr it nl no ptb ro ru
do
   name="exp3"
   name1="exp1"
   echo $name
   echo $name1
   sed -i "s/$name/$name1/g" $value.slurm
   sbatch $value.slurm
done

