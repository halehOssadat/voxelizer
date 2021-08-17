#! /bin/bash
#
#    Submit job as in GPU clusters
#       sbatch run_vox.sh
#
#SBATCH --partition gpu20
#SBATCH --time 3:00:00
#SBATCH --job-name=voxelizer
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres gpu:1
#SBATCH -D /HPS/Controlled_release_ToOp/work/repos/voxelizer/
#SBATCH -o /HPS/Controlled_release_ToOp/work/repos/voxelizer/logs/slurm-%x-%j.log
#SBATCH --mem-per-cpu 50G
#SBATCH --array=1-12

NUMLINES=12
STOP=12
START="$(( $STOP - $(($NUMLINES - 1))))"

echo "START=$START"
echo "STOP=$STOP"

for (( N = $START; N <= $STOP; N++ ))
do
    LINE_In=$(sed -n "$N"p Input.txt)
    LINE_out=$(sed -n "$N"p Output.txt)
    res=$(($(basename $LINE_out) - 1))
    echo "resolution: $res"
    # call your program here
    ./build/bin/voxelizer -r $res -i 3 -n $SLURM_NTASKS $LINE_In $LINE_out -f binvox 
    
done

