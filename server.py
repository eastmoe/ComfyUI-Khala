from __future__ import annotations

from pathlib import Path

from aiohttp import web


ROOT = Path(__file__).resolve().parent
_ROUTES_REGISTERED = False


def register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return

    try:
        from server import PromptServer
    except ImportError:
        return

    routes = PromptServer.instance.routes

    @routes.get("/comfy-khala/locale/{locale}/nodes.json")
    async def get_khala_locale(request):
        locale = request.match_info["locale"].lower()
        if locale != "zh-sn":
            raise web.HTTPNotFound()
        path = ROOT / "locale" / locale / "nodes.json"
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    _ROUTES_REGISTERED = True
