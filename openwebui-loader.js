(function () {
  const DEFAULT_MODEL = 'deepseek-v4-pro';
  const OLD_DEFAULT_MODELS = new Set(['gemini-3.5-flash-agent']);

  const normalizeModelId = (model) => (OLD_DEFAULT_MODELS.has(model) ? DEFAULT_MODEL : model);

  const normalizeUserId = (value) => {
    const safe = String(value || '')
      .trim()
      .replace(/[^a-zA-Z0-9_.@-]+/g, '-')
      .replace(/^[.-]+|[.-]+$/g, '');
    return safe.slice(0, 64);
  };

  const detectOpenWebuiUser = () => {
    const keys = ['user', 'user-info', 'open-webui:user', 'session'];
    for (const key of keys) {
      const raw = localStorage.getItem(key) || sessionStorage.getItem(key);
      if (!raw) continue;
      try {
        const data = JSON.parse(raw);
        const value = data?.email || data?.username || data?.name || data?.id;
        const userId = normalizeUserId(value);
        if (userId) return userId;
      } catch {
        // Ignore unrelated storage values.
      }
    }
    return '';
  };

  const normalizeModels = (models) => {
    if (Array.isArray(models)) {
      const next = models.map(normalizeModelId);
      if (!next.includes(DEFAULT_MODEL)) next.unshift(DEFAULT_MODEL);
      return Array.from(new Set(next));
    }
    return normalizeModelId(models);
  };

  const normalizeSettingsPayload = (payload) => {
    if (!payload || typeof payload !== 'object') return payload;
    const data = Array.isArray(payload) ? payload.slice() : { ...payload };

    if (data.ui && typeof data.ui === 'object') {
      data.ui = { ...data.ui };
      if ('models' in data.ui) data.ui.models = normalizeModels(data.ui.models);
      if ('default_models' in data.ui) data.ui.default_models = normalizeModelId(data.ui.default_models);
      if ('model_order_list' in data.ui) data.ui.model_order_list = normalizeModels(data.ui.model_order_list);
    }
    if (data.settings && typeof data.settings === 'object') {
      data.settings = normalizeSettingsPayload(data.settings);
    }
    if ('models' in data && Array.isArray(data.models)) data.models = normalizeModels(data.models);
    if ('model' in data && typeof data.model === 'string') data.model = normalizeModelId(data.model);

    return data;
  };

  const rewriteStoredValue = (value) => {
    if (typeof value !== 'string' || !value.includes('gemini-3.5-flash-agent')) return value;
    return value.replaceAll('gemini-3.5-flash-agent', DEFAULT_MODEL);
  };

  const scrubStorage = () => {
    [localStorage, sessionStorage].forEach((storage) => {
      storage.removeItem('anysearch.use_kb');
      storage.removeItem('anysearch.kb_enabled');
      storage.removeItem('anysearch.kb');
      for (let i = 0; i < storage.length; i += 1) {
        const key = storage.key(i);
        if (!key) continue;
        const value = storage.getItem(key);
        const next = rewriteStoredValue(value);
        if (next !== value) storage.setItem(key, next);
      }
    });
  };

  const shouldPatchChatRequest = (url, body) => {
    if (!body || typeof body !== 'string') return false;
    const path = String(url || '');
    return (
      path.includes('/api/chat/completions') ||
      path.includes('/api/v1/chat/completions') ||
      path.includes('/chat/completions')
    );
  };

  const patchChatPayload = (payload) => {
    if (!payload || typeof payload !== 'object' || !(payload.messages || payload.model)) return false;
    const userId = detectOpenWebuiUser();
    if (userId) {
      payload.user = userId;
      payload.metadata = {
        ...(payload.metadata || {}),
        anysearch_user_id: userId
      };
    }
    if (OLD_DEFAULT_MODELS.has(payload.model)) payload.model = DEFAULT_MODEL;
    delete payload.use_kb;
    delete payload.kb_enabled;
    delete payload.knowledge_base;
    return true;
  };

  const removeLegacyKbControls = () => {
    const selectors = [
      '#anysearch-kb',
      '.anysearch-kb',
      '[data-anysearch-kb]',
      '[data-testid="anysearch-kb"]',
      '.kb-toggle',
      '#kb-toggle',
      '[name="use_kb"]'
    ];

    document.querySelectorAll(selectors.join(',')).forEach((node) => {
      const root = node.closest('label, button, [role="button"], .anysearch-control, div') || node;
      root.remove();
    });

    document.querySelectorAll('label, button, [role="button"], span, div').forEach((node) => {
      const text = (node.textContent || '').replace(/\s+/g, '');
      if (!text || !text.includes('知识库')) return;
      if (!/(AnySearch|知识库[:：]|启用知识库|使用知识库|知识库开关)/.test(node.textContent || '')) return;
      const root = node.closest('label, button, [role="button"], .anysearch-control') || node;
      root.remove();
    });
  };

  const patchFetch = () => {
    if (window.__anysearchFetchPatched) return;
    window.__anysearchFetchPatched = true;
    const originalFetch = window.fetch.bind(window);

    window.fetch = async (input, init) => {
      let nextInput = input;
      let nextInit = init ? { ...init } : {};
      let url = typeof input === 'string' ? input : input && input.url;
      let body = nextInit.body;

      if (!body && input instanceof Request) {
        try {
          body = await input.clone().text();
          nextInput = new Request(input, { body });
        } catch {
          body = null;
        }
      }

      if (shouldPatchChatRequest(url, body)) {
        try {
          const payload = JSON.parse(body);
          if (patchChatPayload(payload)) nextInit.body = JSON.stringify(payload);
        } catch {
          // Leave non-JSON requests untouched.
        }
      }

      if (body && String(url || '').includes('/api/v1/users/user/settings')) {
        try {
          nextInit.body = JSON.stringify(normalizeSettingsPayload(JSON.parse(body)));
        } catch {
          // Leave non-JSON settings requests untouched.
        }
      }

      const response = await originalFetch(nextInput, nextInit);
      const responseUrl = String(url || response.url || '');
      if (responseUrl.includes('/api/v1/users/user/settings')) {
        try {
          const data = normalizeSettingsPayload(await response.clone().json());
          return new Response(JSON.stringify(data), {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers
          });
        } catch {
          return response;
        }
      }

      return response;
    };
  };

  const patchXhr = () => {
    if (window.__anysearchXhrPatched || !window.XMLHttpRequest) return;
    window.__anysearchXhrPatched = true;
    const originalOpen = window.XMLHttpRequest.prototype.open;
    const originalSend = window.XMLHttpRequest.prototype.send;

    window.XMLHttpRequest.prototype.open = function patchedOpen(method, url, ...rest) {
      this.__anysearchUrl = url;
      return originalOpen.call(this, method, url, ...rest);
    };

    window.XMLHttpRequest.prototype.send = function patchedSend(body) {
      let nextBody = body;
      if (shouldPatchChatRequest(this.__anysearchUrl, body)) {
        try {
          const payload = JSON.parse(body);
          if (patchChatPayload(payload)) nextBody = JSON.stringify(payload);
        } catch {
          // Leave non-JSON requests untouched.
        }
      }
      return originalSend.call(this, nextBody);
    };
  };

  const refresh = () => {
    scrubStorage();
    removeLegacyKbControls();
  };

  scrubStorage();
  removeLegacyKbControls();
  patchFetch();
  patchXhr();
  window.addEventListener('focus', refresh);
  document.addEventListener('DOMContentLoaded', refresh);
  new MutationObserver(removeLegacyKbControls).observe(document.documentElement, {
    childList: true,
    subtree: true
  });
})();
