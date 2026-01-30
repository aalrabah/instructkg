## if co-occurance is zero then we look into clusters co-occurance, and the default is chunk level co-occurance, cluster co-occurance.. 

 then, for the actual LLM judgment (example: concept A is Sorting algorithms, concept B is mergesort):
•⁠  ⁠let's say A and B only co-occur twice. first where A is being defined and B is an assumption, and second where A and B are both examples.
•⁠  ⁠In the prompt, include the following info: what is a concept? what is a role, define each one. then state the stats: A is introduced before B; A is mentioned 10 times before B is first introduced. they co-occur in the same excerpt twice. here are how they co-occur:
•⁠  ⁠A is being defined; B is a concept that is assumed to already be known. here is [1] chunk in which this takes place: [chunk]
•⁠  ⁠A is being demonstrated through an example; B is being demonstrated through an example. here is the [1] chunk in which this takes place: [chunk]

make a judgement on their relationship. here are the two different relationships, partOf and dependsOn, and their respective definitions.
now IF A and B do not ever co-occur in the same chunk, but they do co-occur in the same cluster, do the same thing as above, but state the respective roles that they play and here are the chunks that they play this role in that were placed in the same cluster.


