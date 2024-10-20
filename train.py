input_nc=3
output_nc=3
# Initialize the generators
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = ResnetGenerator(input_nc=input_nc, output_nc=output_nc).to(device)  # Generator A -> B (Photos to Ukiyo-e)
F = ResnetGenerator(input_nc=output_nc, output_nc=input_nc).to(device)  # Generator B -> A (Ukiyo-e to Photos)

# Initialize the discriminators
D_A = NLayerDiscriminator(input_nc=input_nc).to(device)  # Discriminator for Domain A (Photos)
D_B = NLayerDiscriminator(input_nc=output_nc).to(device)  # Discriminator for Domain B (Ukiyo-e)


# Defining optimizer
optimizer_G = optim.Adam(G.parameters(), lr=0.00025, betas=(0.5, 0.999))
optimizer_F = optim.Adam(F.parameters(), lr=0.00025, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.00025, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.00025, betas=(0.5, 0.999))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# Apply the weights initialization function to each model
G.apply(weights_init_normal)
F.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)



# Adversarial loss
adversarial_loss = nn.MSELoss()

# Cycle consistency
cycle_loss = nn.L1Loss()

# Identity loss
identity_loss = nn.L1Loss()

# Loss weights
lambda_cycle = 10.0
lambda_identity = 5.0


batch_size = 4

# Save directory for saving train checkpoints
save_dir = "/content/drive/MyDrive/GAN/weights"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


num_epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


scaler = GradScaler(device='cuda')


def denormalize(tensor):
    tensor = tensor.detach() * 0.5 + 0.5
    return tensor.cpu().float().numpy()


def plot_images(real_A, fake_B, real_B, fake_A, epoch):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].imshow(np.transpose(denormalize(real_A), (1, 2, 0)))
    axs[0].set_title("Real A")
    axs[1].imshow(np.transpose(denormalize(fake_B), (1, 2, 0)))
    axs[1].set_title("Fake B (A -> B)")
    axs[2].imshow(np.transpose(denormalize(real_B), (1, 2, 0)))
    axs[2].set_title("Real B")
    axs[3].imshow(np.transpose(denormalize(fake_A), (1, 2, 0)))
    axs[3].set_title("Fake A (B -> A)")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(f"Epoch {epoch}")
    plt.show()

# Loading checkpoint weight values (only while resuming training)
G.load_state_dict(torch.load('/content/drive/MyDrive/GAN/weights/G_epoch_52.pth'))
F.load_state_dict(torch.load('/content/drive/MyDrive/GAN/weights/F_epoch_52.pth'))
D_A.load_state_dict(torch.load('/content/drive/MyDrive/GAN/weights/D_A_epoch_52.pth'))
D_B.load_state_dict(torch.load('/content/drive/MyDrive/GAN/weights/D_B_epoch_52.pth'))

# Training Loop
for epoch in range(num_epochs):
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True) as pbar:
        total_loss_G = 0
        total_loss_D_A = 0
        total_loss_D_B = 0

        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            optimizer_G.zero_grad()

            with autocast(device_type='cuda'):
                fake_B = G(real_A)  # A -> B
                fake_A = F(real_B)  # B -> A

                recov_A = F(fake_B)  # B -> A -> B
                recov_B = G(fake_A)  # A -> B -> A

                # Generating labels
                valid = torch.ones_like(D_B(fake_B), requires_grad=False).to(device)
                fake = torch.zeros_like(D_B(fake_B), requires_grad=False).to(device)

                loss_GAN_G = adversarial_loss(D_B(fake_B), valid)
                loss_GAN_F = adversarial_loss(D_A(fake_A), valid)

                loss_cycle_A = cycle_loss(recov_A, real_A) * lambda_cycle
                loss_cycle_B = cycle_loss(recov_B, real_B) * lambda_cycle

                loss_identity_A = identity_loss(F(real_A), real_A) * lambda_identity
                loss_identity_B = identity_loss(G(real_B), real_B) * lambda_identity

                loss_G = loss_GAN_G + loss_GAN_F + loss_cycle_A + loss_cycle_B + loss_identity_A + loss_identity_B

            # Backward pass and update generator weights
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            #  Train Discriminators D_A and D_B
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            with autocast(device_type='cuda'):
                loss_real_A = adversarial_loss(D_A(real_A), valid)
                loss_fake_A = adversarial_loss(D_A(fake_A.detach()), fake)
                loss_D_A = (loss_real_A + loss_fake_A) * 0.5

                loss_real_B = adversarial_loss(D_B(real_B), valid)
                loss_fake_B = adversarial_loss(D_B(fake_B.detach()), fake)
                loss_D_B = (loss_real_B + loss_fake_B) * 0.5

            # Backward pass and update discriminator weights
            scaler.scale(loss_D_A).backward()
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizer_D_A)
            scaler.step(optimizer_D_B)
            scaler.update()

            total_loss_G += loss_G.item()
            total_loss_D_A += loss_D_A.item()
            total_loss_D_B += loss_D_B.item()

            pbar.set_postfix({
                'Loss_G': total_loss_G / (i + 1),
                'Loss_D_A': total_loss_D_A / (i + 1),
                'Loss_D_B': total_loss_D_B / (i + 1)
            })
            pbar.update(1)

    avg_loss_G = total_loss_G / len(dataloader)
    avg_loss_D_A = total_loss_D_A / len(dataloader)
    avg_loss_D_B = total_loss_D_B / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Avg Loss_G: {avg_loss_G:.4f}, Avg Loss_D_A: {avg_loss_D_A:.4f}, Avg Loss_D_B: {avg_loss_D_B:.4f}")
    plot_images(real_A[0], fake_B[0], real_B[0], fake_A[0], epoch + 1)

    # Save model parameters
    torch.save(G.state_dict(), os.path.join(save_dir, f'G_epoch_{epoch+52}.pth'))
    torch.save(F.state_dict(), os.path.join(save_dir, f'F_epoch_{epoch+52}.pth'))
    torch.save(D_A.state_dict(), os.path.join(save_dir, f'D_A_epoch_{epoch+52}.pth'))
    torch.save(D_B.state_dict(), os.path.join(save_dir, f'D_B_epoch_{epoch+52}.pth'))




