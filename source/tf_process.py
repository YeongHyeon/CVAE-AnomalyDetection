import os, inspect, time, math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gaussian_sample(batch_size, z_dim, mean=0, sigma=1):

    return np.random.normal(loc=mean, scale=sigma, size=(batch_size, z_dim)).astype(np.float32)

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp
    if(dc == 1):
        canvas_rgb = np.ones((dh*numd, dw*numd, 3)).astype(np.float32)
        canvas_rgb[:, :, 0] = canvas[:, :, 0]
        canvas_rgb[:, :, 1] = canvas[:, :, 0]
        canvas_rgb[:, :, 2] = canvas[:, :, 0]
        canvas = canvas_rgb

    return canvas

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)

def latent_plot(latent, y, n, savename=""):

    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=y, \
        marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def training(sess, saver, neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    summary_writer = tf.compat.v1.summary.FileWriter(PACK_PATH+'/Checkpoint', sess.graph)

    make_dir(path="results")
    result_list = ["tr_latent", "tr_resotring", "tr_latent_walk"]
    for result_name in result_list: make_dir(path=os.path.join("results", result_name))

    start_time = time.time()
    iteration = 0

    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        x_tr, y_tr, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch
        x_restore, z_enc = sess.run([neuralnet.x_hat, neuralnet.z_enc], \
            feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:test_size})
        latent_plot(latent=z_enc, y=y_tr, n=dataset.num_class, savename=os.path.join("results", "tr_latent", "%08d.png" %(epoch)))
        save_img(contents=[x_tr, x_restore, (x_tr-x_restore)**2], \
            names=["Input\n(x)", "Restoration\n(x to x-hat)", "Difference"], \
            savename=os.path.join("results", "tr_resotring", "%08d.png" %(epoch)))

        x_values = np.linspace(-3, 3, test_sq)
        y_values = np.linspace(-3, 3, test_sq)
        z_latents = None
        for y_loc, y_val in enumerate(y_values):
            for x_loc, x_val in enumerate(x_values):
                z_latent = np.reshape(np.array([y_val, x_val]), (1, neuralnet.z_dim))
                if(z_latents is None): z_latents = z_latent
                else: z_latents = np.append(z_latents, z_latent, axis=0)
        x_samples = sess.run(neuralnet.x_sample, \
            feed_dict={neuralnet.z:z_latents, neuralnet.batch_size:test_size})
        plt.imsave(os.path.join("results", "tr_latent_walk", "%08d.png" %(epoch)), dat2canvas(data=x_samples))

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.

            _, summaries = sess.run([neuralnet.optimizer, neuralnet.summaries], \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]}, \
                options=run_options, run_metadata=run_metadata)
            restore, kld, elbo = sess.run([neuralnet.mean_restore, neuralnet.mean_kld, neuralnet.loss], \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})
            summary_writer.add_summary(summaries, iteration)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  Restore:%.3f, KLD:%.3f, Total:%.3f" \
            %(epoch, epochs, iteration, restore, kld, elbo))
        summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch)

def test(sess, saver, neuralnet, dataset, batch_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("\nTest...")

    while(True):
        x_te, y_te, terminator = dataset.next_test(batch_size) # y_te does not used in this prj.
        if(terminator): break
