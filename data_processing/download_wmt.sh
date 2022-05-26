#!/bin/bash

mkdir -p seq
cd seq

#Getting common crawl data for training
wget --trust-server-names http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
tar zxvf training-parallel-commoncrawl.tgz
ls | grep -v 'commoncrawl.de-en.[de,en]' | xargs rm


#Getting europarl data for training
wget --trust-server-names http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
tar zxvf training-parallel-europarl-v7.tgz
cd training && ls | grep -v 'europarl-v7.de-en.[de,en]' | xargs rm
cd .. && mv training/europarl* . && rm -r training training-parallel-europarl-v7.tgz


#concat train files
cat commoncrawl.de-en.en europarl-v7.de-en.en > train_orig.src
cat commoncrawl.de-en.de europarl-v7.de-en.de > train_orig.trg

#downsize train files
awk '{ if (NR % 10 == 0) print $0; }' train_orig.src > train.src
awk '{ if (NR % 10 == 0) print $0; }' train_orig.trg > train.trg
rm train_orig.src train_orig.trg


#Getting Valid and Test datset
wget --trust-server-names https://www.statmt.org/wmt14/dev.tgz
wget --trust-server-names https://www.statmt.org/wmt14/test-filtered.tgz
tar zxvf dev.tgz && tar zxvf test-filtered.tgz

mv dev/newstest2013.en valid.src
mv dev/newstest2013.de valid.trg

mv test/newstest2014-deen-src.en.sgm .
mv test/newstest2014-deen-ref.de.sgm .

wget -nc https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/ems/support/input-from-sgm.perl
perl input-from-sgm.perl < newstest2014-deen-src.en.sgm > test.src
perl input-from-sgm.perl < newstest2014-deen-ref.de.sgm > test.trg

rm -rf dev test *sgm *perl *tgz common* euro*