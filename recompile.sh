function usage {
   cat << EOF
Usage: recompile.sh
Compiles tabular and core modules

Usage: recompile.sh -t
Compiles tabular module

Usage: recompile.sh -c
Compiles core module
EOF
   exit 1
}

if [ $# == 0 ]; then
    cd ~/autogluon/tabular/
    pip install -e .
    cd ~/autogluon/core
    pip install -e .
    cd ~/autogluon
fi

if [ $# -ne 1 ]; then
   usage
fi

if [ "$1" == "-t" ]; then
    cd ~/autogluon/tabular/
    pip install -e .
    cd ~/autogluon
fi

if [ "$1" == "-c" ]; then
    cd ~/autogluon/core/
    pip install -e .
    cd ~/autogluon
fi
