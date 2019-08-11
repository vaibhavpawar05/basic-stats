### Puzzles

A worker’s legal code specifies as a holiday any day during which at least one worker in a certain factory has a birthday. All other days are working days. How many workers (n) must the factory employ so that the expected number of working man-days is maximized during the year?

Let Xi be an rv that equals 1 if none of the n workers has his birthday on this specific day whereas Xi = 0 if at least one worker has his birthday on that day. With n workers, how likely is it that Xi = 1? Clearly, for any individual worker, the probability not to have his birthday on day i equals 364/365. Therefore, with independent birthdays, the probability P(Xi = 1) that no worker has his birthday on day i is equal to (364/365)^n. This implies that the expectation E[Xi] = P(Xi = 1) = (364/365)^n

Now, define the rv S = sum(Xi).
E[S] = sum(E[Xi]) = 365*(364/365)^n
Now, f(n) = n*E[S], because in expectation the n workers will have to work on E[S] days, yielding n*E[S] man-days. Thus, f(n) = n*365*(364/365)^n

An urn contains six balls — three red and three blue. One of these balls — let us call it ball A — is selected at random and permanently removed from the urn without the color of this ball being shown to an observer. This observer may now draw successively — at random and with replacement — a number of individual balls (one at a time) from among the five remaining balls, so as to form a noisy impression about the ratio of red vs. blue balls that remained in the urn after A was removed.
Peter may draw a ball six times, and each time the ball he draws turns out to be red. Paula may draw a ball 600 times; 303 times she draws a red ball, and 297 times a blue ball. Clearly, both will tend to predict that ball A was probably blue. Which of them — if either — has the stronger empirical evidence for his/her prediction?

Use Bayes theorem - P(A=blue|6 times r)

Peter and Paula are given copies of the same text for independent proof- reading. Peter finds 20 errors, and Paula finds 15 errors, of which 10 were found by Peter as well. Estimate the number of errors remaining in the text that have not been detected by either Peter or Paula.

Assume that Peter and Paula independently detect any given error with probabilities pA,pB, and denote the (unknown) total number of errors in the text as n. Denote as nA,nB, and nAB, respectively, the number of errors detected by Peter, by Paula, and by both.

nAB = (nA*nB)/n

So n = (nA*nB)/nAB = (20*15)/10 = 30. So not found = 30 - 25 = 5

How to sample a point uniformly from a circle? How to do that in polar co-ordinates?
