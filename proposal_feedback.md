Score: 5/8

Apologies for the delay in getting this feedback to you. This looks fantastic, but I’m asking for a few more details below. You have a week from when you receive this until you need to submit a revised version. Please feel free to reach out with specific questions you have about this feedback, and please find a time to discuss it with your project mentor.

Thanks for the detailed discussion of how to construct a smaller dataset. You definitely shouldn’t *limit* yourself to a dataset of ~150 notes, the idea of a 10MB dataset is just to start there to make sure that your model runs and actually learns something. It’s much easier to debug your model code when every epoch takes two minutes rather than two hours.

Essential goals:
1.	I think the VQ-VAE is going to be easier and faster than a diffusion model, so I’d encourage you to start there and then expand to a diffusion model as either a desired or stretch goal.
2.	Are you starting from an existing VQ-VAE implementation or writing this code yourself? I’m assuming you’re using existing code, but I don’t see a link to that codebase. Writing your own code is fine too, but it’s going to be a lot of work and may prevent you from getting to some of your more ambitious goals. 
3.	It would be great to achieve low reconstruction loss, but can you provide a bit of detail as to how you will go about trying to improve your VQ-VAE’s performance on this metric? What hyperparameters will you adjust and why?

Desired goals:
1.	Demonstrating the compression abilities of your model would be very cool, but I’m not entirely sure how you plan to evaluate this. Getting the compressed version from your model seems straightforward, but (a) is it trivial to link separate notes back together to create the original audio? It seems plausible that doing so would introduce a lot of artifacts at the transitions between notes, which your model might not be explicitly designed to prevent, (b) what (existing compression) methods will you compare this against? (c) how will you measure the quality of the decompressed audio? Just qualitatively, or with some sort of metric?
2.	This is a very ambitious desired goal that I think requires several improvements to your model. First, the question above regarding how to connect sampled notes into a melody is again central – how do you plan to do that? Second, adding in style disentanglement in your autoencoder might be difficult. How do you plan to achieve this? What kind of stylistic labels do you have for your existing dataset?

Stretch goals:
1.	These are both extremely ambitious! This is great, but it would be helpful for you to describe a smaller version of each that you think could be completed by the two of you in a single week. For example, you might try to collect a small new dataset of music created by beginners along with the actual sheet music and then fine-tune a classifier on top of your model to predict that sheet music representation.
2.	Similarly for your second stretch goal – this sounds to me like a months or years-long research/software project; what’s the tiniest version of this that you could complete in a week?
3.	Your third goal regarding producing simple melodies seems to be a prerequisite for the desired goals above; is it possible to do those without having completed this?
