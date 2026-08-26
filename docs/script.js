const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const header = $('[data-header]');
const progress = $('.scroll-progress span');
const updateScroll = () => {
  const y = window.scrollY;
  header.classList.toggle('scrolled', y > 20);
  const available = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.width = `${available > 0 ? (y / available) * 100 : 0}%`;
};
updateScroll();
window.addEventListener('scroll', updateScroll, { passive: true });

const navToggle = $('.nav-toggle');
const nav = $('#site-nav');
navToggle.addEventListener('click', () => {
  const open = nav.classList.toggle('open');
  navToggle.setAttribute('aria-expanded', String(open));
});
$$('.site-nav a').forEach((link) => link.addEventListener('click', () => {
  nav.classList.remove('open');
  navToggle.setAttribute('aria-expanded', 'false');
}));

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
$$('.reveal').forEach((element) => observer.observe(element));

const scaleDescriptions = {
  1: 'Maximum local detail for precise boundaries and compact objects.',
  3: 'Short-range context links nearby structures and reduces local ambiguity.',
  14: 'Long-range observation reveals global layout and semantic continuity.'
};
const frustum = $('[data-frustum]');
const distance = $('[data-distance]');
const scaleCopy = $('[data-scale-copy]');
frustum.dataset.activeScale = '1';
$$('[data-scale]').forEach((button) => button.addEventListener('click', () => {
  const scale = button.dataset.scale;
  $$('[data-scale]').forEach((item) => {
    const active = item === button;
    item.classList.toggle('active', active);
    item.setAttribute('aria-pressed', String(active));
  });
  frustum.dataset.activeScale = scale;
  distance.textContent = `${scale}×`;
  scaleCopy.textContent = scaleDescriptions[scale];
}));

const galleryItems = [
  {
    title: 'GID',
    src: 'assets/qualitative-gid.png',
    alt: 'Qualitative comparison on GID',
    copy: 'Large-scale land-cover regions stay coherent without sacrificing local boundaries.'
  },
  {
    title: 'FBPS',
    src: 'assets/qualitative-fbps.png',
    alt: 'Qualitative comparison on FBPS',
    copy: 'Dense multi-class urban scenes benefit from context that reaches far beyond a single patch.'
  }
];
let galleryIndex = 0;
const galleryFrame = $('[data-gallery-image]');
const galleryImage = $('img', galleryFrame);
const showGalleryItem = (nextIndex) => {
  galleryIndex = (nextIndex + galleryItems.length) % galleryItems.length;
  const item = galleryItems[galleryIndex];
  galleryFrame.classList.add('switching');
  window.setTimeout(() => {
    galleryImage.src = item.src;
    galleryImage.alt = item.alt;
    galleryFrame.dataset.lightbox = item.src;
    galleryFrame.setAttribute('aria-label', `Open ${item.title} qualitative results`);
    $('[data-gallery-title]').textContent = item.title;
    $('[data-gallery-kicker]').textContent = `Dataset 0${galleryIndex + 1} / 02`;
    $('[data-gallery-copy]').textContent = item.copy;
    $$('[data-gallery-dot]').forEach((dot, index) => dot.classList.toggle('active', index === galleryIndex));
    galleryFrame.classList.remove('switching');
  }, 160);
};
$('[data-gallery-prev]').addEventListener('click', () => showGalleryItem(galleryIndex - 1));
$('[data-gallery-next]').addEventListener('click', () => showGalleryItem(galleryIndex + 1));
$$('[data-gallery-dot]').forEach((dot) => dot.addEventListener('click', () => showGalleryItem(Number(dot.dataset.galleryDot))));
galleryFrame.closest('.gallery').addEventListener('keydown', (event) => {
  if (event.key === 'ArrowLeft') showGalleryItem(galleryIndex - 1);
  if (event.key === 'ArrowRight') showGalleryItem(galleryIndex + 1);
});

$$('[data-ablation]').forEach((tab) => tab.addEventListener('click', () => {
  $$('[data-ablation]').forEach((item) => {
    const active = item === tab;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', String(active));
  });
  $$('.ablation-panel').forEach((panel) => {
    const active = panel.id === `ablation-${tab.dataset.ablation}`;
    panel.hidden = !active;
    panel.classList.toggle('active', active);
  });
}));

const dialog = $('.lightbox');
const dialogImage = $('img', dialog);
$$('[data-lightbox]').forEach((trigger) => trigger.addEventListener('click', () => {
  const sourceImage = $('img', trigger);
  dialogImage.src = trigger.dataset.lightbox;
  dialogImage.alt = sourceImage?.alt || 'Paper figure';
  dialog.showModal();
}));
$('.lightbox-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => {
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
});

const copyButton = $('[data-copy-citation]');
copyButton.addEventListener('click', async () => {
  const citation = $('[data-citation]').textContent;
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = 'Copied';
  } catch {
    const range = document.createRange();
    range.selectNodeContents($('[data-citation]'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.textContent = 'Selected';
  }
  window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
});
