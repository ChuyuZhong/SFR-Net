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

const demoCard = $('.demo-card');
const demoViewports = $$('[data-zoom-viewport]');
const demoImages = [$('[data-demo-original]'), $('[data-demo-result]')];
const demoRange = $('[data-demo-range]');
const demoZoomOutput = $('[data-demo-zoom]');
let demoState = { zoom: 1, x: 0, y: 0 };
let dragState = null;

const applyDemoTransform = () => {
  const transform = 'translate3d(' + demoState.x + 'px,' + demoState.y + 'px,0) scale(' + demoState.zoom + ')';
  demoImages.forEach((image) => { image.style.transform = transform; });
  demoRange.value = String(Math.round(demoState.zoom * 100));
  demoZoomOutput.value = Math.round(demoState.zoom * 100) + '%';
  demoZoomOutput.textContent = Math.round(demoState.zoom * 100) + '%';
};

const clampDemoPan = () => {
  const rect = demoViewports[0].getBoundingClientRect();
  const maxX = Math.max(0, rect.width * (demoState.zoom - 1) / 2);
  const maxY = Math.max(0, rect.height * (demoState.zoom - 1) / 2);
  demoState.x = Math.max(-maxX, Math.min(maxX, demoState.x));
  demoState.y = Math.max(-maxY, Math.min(maxY, demoState.y));
};

const setDemoZoom = (nextZoom) => {
  demoState.zoom = Math.max(1, Math.min(10, nextZoom));
  if (demoState.zoom === 1) { demoState.x = 0; demoState.y = 0; }
  clampDemoPan();
  applyDemoTransform();
  demoCard.classList.add('interacted');
};

const resetDemoView = () => {
  demoState = { zoom: 1, x: 0, y: 0 };
  applyDemoTransform();
};

demoRange.addEventListener('input', () => setDemoZoom(Number(demoRange.value) / 100));
$('[data-demo-minus]').addEventListener('click', () => setDemoZoom(demoState.zoom - .25));
$('[data-demo-plus]').addEventListener('click', () => setDemoZoom(demoState.zoom + .25));
$('[data-demo-reset]').addEventListener('click', resetDemoView);

demoViewports.forEach((viewport) => {
  viewport.addEventListener('wheel', (event) => {
    event.preventDefault();
    setDemoZoom(demoState.zoom + (event.deltaY < 0 ? .2 : -.2));
  }, { passive: false });

  viewport.addEventListener('pointerdown', (event) => {
    dragState = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, imageX: demoState.x, imageY: demoState.y };
    viewport.setPointerCapture(event.pointerId);
    demoViewports.forEach((item) => item.classList.add('dragging'));
    demoCard.classList.add('interacted');
  });

  viewport.addEventListener('pointermove', (event) => {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    demoState.x = dragState.imageX + event.clientX - dragState.startX;
    demoState.y = dragState.imageY + event.clientY - dragState.startY;
    clampDemoPan();
    applyDemoTransform();
  });

  const endDrag = (event) => {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    dragState = null;
    demoViewports.forEach((item) => item.classList.remove('dragging'));
  };
  viewport.addEventListener('pointerup', endDrag);
  viewport.addEventListener('pointercancel', endDrag);
});

$$('[data-demo-sample]').forEach((button) => button.addEventListener('click', () => {
  const sample = Number(button.dataset.demoSample);
  $$('[data-demo-sample]').forEach((item) => {
    const active = item === button;
    item.classList.toggle('active', active);
    item.setAttribute('aria-pressed', String(active));
  });
  demoImages[0].src = 'assets/demo/demo-' + sample + '-original.jpg';
  demoImages[1].src = 'assets/demo/demo-' + sample + '-result.png';
  demoImages[0].alt = 'Ultra-wide remote sensing source image, scene ' + sample;
  demoImages[1].alt = 'SFR-Net semantic segmentation result, scene ' + sample;
  $('[data-demo-name]').textContent = 'FBPS · Scene 0' + sample;
  resetDemoView();
}));

for (let sample = 2; sample <= 5; sample += 1) {
  const original = new Image();
  const result = new Image();
  original.src = 'assets/demo/demo-' + sample + '-original.jpg';
  result.src = 'assets/demo/demo-' + sample + '-result.png';
}

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
