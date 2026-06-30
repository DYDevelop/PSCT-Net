const copyButtons = document.querySelectorAll('[data-copy-target]');

copyButtons.forEach((button) => {
  button.addEventListener('click', async () => {
    const targetId = button.getAttribute('data-copy-target');
    const target = document.getElementById(targetId);
    if (!target) return;

    const text = target.innerText.trim();
    try {
      await navigator.clipboard.writeText(text);
      const oldText = button.textContent;
      button.textContent = 'Copied';
      setTimeout(() => { button.textContent = oldText; }, 1400);
    } catch (error) {
      button.textContent = 'Copy failed';
      setTimeout(() => { button.textContent = 'Copy BibTeX'; }, 1400);
    }
  });
});
