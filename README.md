# easyml-examples
#打包工程
mvn clean assembly:assembly

#生成可执行的zip压缩包
mkdir target/easyml-program
cp -rp target/easyml-examples-0.01-SNAPSHOT-jar-with-dependencies.jar target/easyml-program/easyml-examples-0.01-SNAPSHOT.jar
cd target
7z a -tzip easyml-program.zip easyml-program
cp -rp easyml-program.zip /d/work/easyml/
cd ..