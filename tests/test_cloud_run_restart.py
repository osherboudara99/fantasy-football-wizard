from pipeline.cloud_run_restart import restart_cloud_run_service


class _FakeTemplate:
    def __init__(self):
        self.annotations = {}


class _FakeService:
    def __init__(self):
        self.template = _FakeTemplate()


class _FakeOperation:
    def __init__(self):
        self.resolved = False

    def result(self):
        self.resolved = True


class _FakeServicesClient:
    def __init__(self, service):
        self._service = service
        self.get_service_calls = []
        self.update_service_calls = []
        self.last_operation: _FakeOperation | None = None

    def get_service(self, name):
        self.get_service_calls.append(name)
        return self._service

    def update_service(self, service):
        self.update_service_calls.append(service)
        self.last_operation = _FakeOperation()
        return self.last_operation


def test_restart_cloud_run_service_updates_the_named_service():
    service = _FakeService()
    client = _FakeServicesClient(service)

    restart_cloud_run_service("my-project", "us-central1", "ffw-api", client=client)

    assert client.get_service_calls == ["projects/my-project/locations/us-central1/services/ffw-api"]
    assert client.update_service_calls == [service]


def test_restart_cloud_run_service_touches_an_annotation_to_force_a_new_revision():
    service = _FakeService()
    client = _FakeServicesClient(service)

    restart_cloud_run_service("my-project", "us-central1", "ffw-api", client=client)

    assert "ffw-refresh-ts" in service.template.annotations


def test_restart_cloud_run_service_waits_for_the_update_operation_to_complete():
    service = _FakeService()
    client = _FakeServicesClient(service)

    restart_cloud_run_service("my-project", "us-central1", "ffw-api", client=client)

    assert client.last_operation.resolved is True
