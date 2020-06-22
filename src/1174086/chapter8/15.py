            # Write losses to Tensorboard
            write_log(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
            write_log(tensorboard, 'd_loss', np.mean(dis_losses), epoch)

